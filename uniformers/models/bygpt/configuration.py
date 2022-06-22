from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from inspect import signature


class ByGPTConfig(GPT2Config):
    model_type = "bygpt"
    def __init__(
        self,
        bos_token_id=0,
        eos_token_id=1,
        unk_token_id=2,
        pad_token_id=3,
        vocab_size=2**8 + 4 + 125,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            **kwargs
        )

    @staticmethod
    def generate_from_pretrained_gpt2(pretrained_model_name_or_path, **kwargs):
        """
        Approximately equalize number of parameters of byGPT and GPT-2 by
        increasing hidden size and feed forward dimensionality (like byT5).
        """
        vocab_size = signature(ByGPTConfig.__init__).parameters["vocab_size"].default
        match pretrained_model_name_or_path:
            case "gpt2":  # 124.4M parameters
                n_embd, n_inner = 924, 3703
            case "gpt2-medium":  # 354.8M parameters
                n_embd, n_inner = 1104, 4454
            case "gpt2-large":  # 774.0M parameters
                n_embd, n_inner = 1320, 5479
            case "gpt2-xl":  # 1557.6M parameters
                n_embd, n_inner = 1625, 6714
            case _:
                raise ValueError

        config = ByGPTConfig.from_pretrained(
            pretrained_model_name_or_path,
            n_embd=n_embd,
            n_inner=n_inner,
            vocab_size=vocab_size,
            **kwargs
        )
        return config

    @staticmethod
    def generate_from_pretrained_byt5(pretrained_model_name_or_path, **kwargs):
        raise NotImplementedError # TODO
