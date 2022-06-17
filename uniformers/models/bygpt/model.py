from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel,
    GPT2DoubleHeadsModel,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
)
from .configuration import ByGPTConfig


class ByGPTModel(GPT2Model):
    model_type = "bygpt"
    config_class = ByGPTConfig


class ByGPTLMHeadModel(GPT2LMHeadModel):
    model_type = "bygpt"
    config_class = ByGPTConfig


class ByGPTDoubleHeadsModel(GPT2DoubleHeadsModel):
    model_type = "bygpt"
    config_class = ByGPTConfig


class ByGPTForSequenceClassification(GPT2ForSequenceClassification):
    model_type = "bygpt"
    config_class = ByGPTConfig


class ByGPTForTokenClassification(GPT2ForTokenClassification):
    model_type = "bygpt"
    config_class = ByGPTConfig
