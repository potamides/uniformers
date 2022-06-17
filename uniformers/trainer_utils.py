from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .trainer import LMTrainer


def load_or_train_lm(
    output_dir,
    config,
    modelclass=AutoModelForCausalLM,
    tokenizerclass=AutoTokenizer,
    trainer=LMTrainer,
    test_run=False,
):
    try:
        model, tokenizer = modelclass.from_pretrained(
            output_dir
        ), tokenizerclass.from_pretrained(output_dir)
    except EnvironmentError:
        model, tokenizer = modelclass(config), tokenizerclass()
        trainer = trainer(model, tokenizer, "models/bygpt", test_run=test_run)
        trainer.train()
        trainer.save_model()
        trainer.save_state()
    return model, tokenizer
