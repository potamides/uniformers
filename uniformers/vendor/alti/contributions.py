import torch
import torch.nn.functional as F
from functools import partial
import collections
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()

        self.model = model

        self.num_attention_heads = self.model.config.num_attention_heads
        try:
            self.attention_head_size = self.model.config.d_kv
            self.all_head_size = self.model.config.d_model
        except AttributeError:
            self.attention_head_size = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

    def save_activation(self,name, _, inp, out):
        self.func_inputs[name].append(inp)
        self.func_outputs[name].append(out)

    def clean_hooks(self):
        for k, _ in self.handles.items():
            self.handles[k].remove()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_contributions(self, hidden_states_model, attentions, func_inputs, func_outputs):
        #   hidden_states_model: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
        #   attentions: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)

        model_importance_list = []
        transformed_vectors_norm_list = []
        transformed_vectors_list = []
        resultants_list = []
        contributions_data = {}

        try:
            num_layers = self.model.config.n_layers
        except:
            num_layers = self.model.config.num_hidden_layers

        for layer in range(num_layers):
            hidden_states = hidden_states_model[layer]
            attention_probs = attentions[layer]

            #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
            #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
            #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

            if self.model.config.model_type == 'bert':
                value_layer = self.transpose_for_scores(func_outputs[self.model.config.model_type + '.encoder.layer.' + str(layer) + '.attention.self.value'][0])
                dense = self.model.bert.encoder.layer[layer].attention.output.dense
                LayerNorm = self.model.bert.encoder.layer[layer].attention.output.LayerNorm
                pre_ln_states = func_inputs[self.model.config.model_type +'.encoder.layer.' + str(layer) + '.attention.output.LayerNorm'][0][0]
            elif self.model.config.model_type == 'distilbert':
                value_layer = self.transpose_for_scores(func_outputs[self.model.config.model_type + '.transformer.layer.' + str(layer) + '.attention.v_lin'][0])
                dense = self.model.distilbert.transformer.layer[layer].attention.out_lin
                LayerNorm = self.model.distilbert.transformer.layer[layer].sa_layer_norm
                pre_ln_states = func_inputs[self.model.config.model_type +'.transformer.layer.' + str(layer) + '.sa_layer_norm'][0][0]
            elif self.model.config.model_type == 'roberta':
                value_layer = self.transpose_for_scores(func_outputs[self.model.config.model_type + '.encoder.layer.' + str(layer) + '.attention.self.value'][0])
                dense = self.model.roberta.encoder.layer[layer].attention.output.dense
                LayerNorm = self.model.roberta.encoder.layer[layer].attention.output.LayerNorm
                pre_ln_states = func_inputs[self.model.config.model_type +'.encoder.layer.' + str(layer) + '.attention.output.LayerNorm'][0][0]
            elif self.model.config.model_type == 'bygpt5':
                value_layer = self.transpose_for_scores(func_outputs['decoder.block.' + str(layer) + '.layer.0.SelfAttention.v'][0])
                dense = self.model.decoder.block[layer].layer[0].SelfAttention.o
                LayerNorm = self.model.decoder.block[layer].layer[0].layer_norm
                pre_ln_states = func_inputs['decoder.block.' + str(layer) + '.layer.0.layer_norm'][0][0]
            elif self.model.config.model_type == 'gpt2':
                _attn = self.model.transformer.h[layer].attn
                *_, value_layer = func_outputs['transformer.h.' + str(layer) + '.attn.c_attn'][0].split(_attn.split_size, dim=2)
                value_layer = _attn._split_heads(value_layer, _attn.num_heads, _attn.head_dim)
                dense = _attn.c_proj
                LayerNorm = self.model.transformer.h[layer].ln_2
                pre_ln_states = func_inputs['transformer.h.' + str(layer) + '.ln_2'][0][0]
            else:
                raise ValueError("Model not supported!")

            # VW_O
            dense_bias = dense.bias if dense.bias is not None else torch.zeros((dense.weight.shape[0]))
            dense = dense.weight.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer, dense) #(batch, num_heads, seq_length, all_head_size)

            # AVW_O
            # (batch, num_heads, seq_length, seq_length, all_head_size)
            #print('transformed_layer', transformed_layer.size())
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer)

            # Sum each weighted vectors αf(x) over all heads:
            # (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1) # sum over heads

            # Make residual matrix (batch, seq_length, seq_length, all_head_size)
            hidden_shape = hidden_states.size()
            device = hidden_states.device
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1]).to(device), hidden_states)

            # AVW_O + residual vectors -> (batch,seq_len,seq_len,embed_dim)
            residual_weighted_layer = summed_weighted_layer + residual

            # consider layernorm
            ln_weight = LayerNorm.weight.data
            try:

                ln_eps = LayerNorm.variance_epsilon
                ln_bias = 0
            except AttributeError:
                ln_eps = LayerNorm.eps
                ln_bias = LayerNorm.bias

            def l_transform(x, w_ln):
                '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
                ln_param_transf = torch.diag(w_ln)
                ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
                    1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

                out = torch.einsum(
                    '... e , e f , f g -> ... g',
                    x,
                    ln_mean_transf,
                    ln_param_transf
                )
                return out

            # Transformed vectors T_i(x_j)
            transformed_vectors = l_transform(residual_weighted_layer, ln_weight) # (batch, seq_len, seq_len, all_head_size)

            # Output vectors 1 per source token
            attn_output = transformed_vectors.sum(dim=2) #(batch,seq_len,all_head_size)

            # Lb_O
            dense_bias_term = l_transform(dense_bias, ln_weight)

            # y_i
            ln_std_coef = 1/(pre_ln_states + ln_eps).std(-1).view(1, -1, 1) # (batch,seq_len,1)
            resultant = (attn_output + dense_bias_term)*ln_std_coef + ln_bias

            transformed_vectors_std = l_transform(residual_weighted_layer, ln_weight)*ln_std_coef.unsqueeze(-1)
            transformed_vectors_norm_std = torch.norm(transformed_vectors_std, dim=-1) # (batch, seq_len, seq_len)

            importance_matrix = -F.pairwise_distance(transformed_vectors_std, resultant.unsqueeze(2),p=1)

            model_importance_list.append(torch.squeeze(importance_matrix).cpu().detach())
            transformed_vectors_norm_list.append(torch.squeeze(transformed_vectors_norm_std).cpu().detach())
            transformed_vectors_list.append(torch.squeeze(transformed_vectors_std).cpu().detach())
            resultants_list.append(torch.squeeze(resultant).cpu().detach())

        contributions_model = torch.stack(model_importance_list)
        transformed_vectors_norm_model = torch.stack(transformed_vectors_norm_list)
        transformed_vectors_model = torch.stack(transformed_vectors_list)
        resultants_model = torch.stack(resultants_list)

        contributions_data['contributions'] = contributions_model
        contributions_data['transformed_vectors'] = transformed_vectors_model
        contributions_data['transformed_vectors_norm'] = transformed_vectors_norm_model
        contributions_data['resultants'] = resultants_model

        return contributions_data

    def get_prediction(self, input_model):
        with torch.no_grad():
            output = self.model(input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits']

            return prediction_scores


    def __call__(self,input_model):
        with torch.no_grad():
            self.handles = {}
            for name, module in self.model.named_modules():
                self.handles[name] = module.register_forward_hook(partial(self.save_activation, name))

            self.func_outputs = collections.defaultdict(list)
            self.func_inputs = collections.defaultdict(list)

            output = self.model(**input_model, output_hidden_states=True, output_attentions=True)
            prediction_scores = output['logits']
            hidden_states = output['hidden_states']
            attentions = output['attentions']

            contributions_data = self.get_contributions(hidden_states, attentions, self.func_inputs, self.func_outputs)

            # Clean forward_hooks dictionaries
            self.clean_hooks()
            return prediction_scores, hidden_states, attentions, contributions_data
