#!/usr/bin/env -S python -m torch.distributed.launch --nproc_per_node gpu
from uniformers.models.bygpt5 import ByGPT5Config, ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.trainer import LMTrainer
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

set_verbosity_info()
enable_explicit_format()

base_model = "google/byt5-small"
output_dir = "data/models/bygpt5-small"

try:
    model, tokenizer = ByGPT5LMHeadModel.from_pretrained(
        output_dir
    ), ByGPT5Tokenizer.from_pretrained(output_dir)
except EnvironmentError:
    config = ByGPT5Config.from_pretrained(base_model)
    model, tokenizer = ByGPT5LMHeadModel(config), ByGPT5Tokenizer()
    trainer = LMTrainer(model, tokenizer, "models/bygpt", test_run=True)
    trainer.train()
    trainer.save_model()
    trainer.save_state()
