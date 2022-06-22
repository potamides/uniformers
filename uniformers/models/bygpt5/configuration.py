from transformers.models.t5.configuration_t5 import T5Config


class ByGPT5Config(T5Config):
    model_type = "bygpt5"
    pass
