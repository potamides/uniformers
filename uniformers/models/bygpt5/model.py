from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils.logging import get_logger
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from .configuration import ByGPT5Config

logger = get_logger("transformers")
# weights which we don't use in our decoder only variant
_bygpt5_keys_to_ignore_on_load_unexpected = [
    # r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    r"encoder.*",
    r"decoder\.block\.\d+\.layer\.1\.layer_norm\.weight",
    r"decoder\.block\.\d+\.layer\.1\.EncDecAttention\.[qkov]\.weight",
]


class ByGPT5Block(T5Block):
    """
    Custom T5Block which does not instantiate T5LayerCrossAttention (which we
    don't need) and which throws errors during parallel training.
    """

    def __init__(self, config, has_relative_attention_bias=False):
        # call __init__ of grandparent (parent contains the code we don't want)
        super(T5Block, self).__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )

        if self.is_decoder:
            # add identity instead of cross attention so that weights are
            # loaded correctly
            self.layer.append(nn.Identity())

        self.layer.append(T5LayerFF(config))

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        # hack: add dummy cross attention to prevent index error in T5Stack
        return outputs + (None, None)


class ByGPT5Stack(T5Stack):
    """
    Overwrite T5Stack to use our custom T5Block class.
    """

    def __init__(self, config, embed_tokens=None):
        # call __init__ of grandparent (parent contains the code we don't want)
        super(T5Stack, self).__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                ByGPT5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        # set cross attention to None, as it doesn't exist in our model
        output.cross_attentions = None

        return output

class ByGPT5Model(ByGPT5Stack):
    model_type = "bygpt5"
    config_class = ByGPT5Config
    _keys_to_ignore_on_load_unexpected = _bygpt5_keys_to_ignore_on_load_unexpected

    def __init__(self, config: ByGPT5Config):
        config = deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        config.num_layers = config.num_decoder_layers
        super().__init__(config)


class ByGPT5LMHeadModel(T5PreTrainedModel):
    model_type = "bygpt5"
    config_class = ByGPT5Config
    _keys_to_ignore_on_load_missing = [
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = _bygpt5_keys_to_ignore_on_load_unexpected

    def __init__(self, config: ByGPT5Config | PretrainedConfig):
        config = deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        config.num_layers = config.num_decoder_layers
        super().__init__(config)

        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = ByGPT5Stack(deepcopy(config), self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.decoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.decoder.block))
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.decoder.deparallelize()
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], CausalLMOutputWithCrossAttentions]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            if input_ids is not None:
                input_ids = input_ids.to(self.decoder.first_device) # pyright: ignore
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device) # pyright: ignore

        # Decode
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            past_key_values=past_key_values,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            self.lm_head = self.lm_head.to(self.decoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape # pyright: ignore
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, use_cache=None, **kwargs):
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        })

        return model_inputs
