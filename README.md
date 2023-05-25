# Uniformers: ByGPT5 & Friends
[![arXiv](https://img.shields.io/badge/arXiv-2212.10474-B31B1B)](https://arxiv.org/abs/2212.10474)
[![Semantic Scholar](https://img.shields.io/badge/Semantic_Scholar-254877406-0f3875)](https://www.semanticscholar.org/paper/ByGPT5%3A-End-to-End-Style-conditioned-Poetry-with-Belouadi-Eger/11ddb0953eae196dab339bfdc117221594cf945e)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ast-seDV6_pSzCvpGapNfTV_qDe0KRdF)

[Uniformers](https://github.com/potamides/uniformers) (short for _**Uni**versal
Coded Character Set Trans**formers**_) is a library for token-free language
modeling. In particular, it contains the reference implementation of _**ByGPT5:
End-to-End Style-conditioned Poetry Generation with Token-free Language
Models**_. ByGPT5 is a token-free decoder-only transformer that excels at
character-level tasks such as style-conditioned poetry generation. 

* :scroll: Read our [paper](https://arxiv.org/abs/2212.10474) on ByGPT5 for details.
* :feather: An interactive demo for poetry generation is [available](https://colab.research.google.com/drive/1Ast-seDV6_pSzCvpGapNfTV_qDe0KRdF).
* :bulb: If you make use of this library in your work please [cite](CITATION.cff) it.

## Installation
If you want to use this project as a library you can install it as a regular
package using [pip](https://pip.pypa.io/en/stable):
```sh
pip install 'git+https://github.com/potamides/uniformers.git#egg=uniformers'
```
If your goal is to run the included [examples](examples) (e.g., to reproduce
results) clone the repository and install it in editable mode:
 ```sh
git clone https://github.com/potamides/uniformers
pip install -e uniformers[examples]
 ```

## Usage
Uniformers builds upon the
[transformers](https://github.com/huggingface/transformers/) library and can be
used very similarly.
```python
from torch import device
from transformers.pipelines.text_generation import TextGenerationPipeline

from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer

prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

pipeline = TextGenerationPipeline(
    model=ByGPT5LMHeadModel.from_pretrained("nllg/bygpt5-medium-en"),
    tokenizer=ByGPT5Tokenizer.from_pretrained("nllg/bygpt5-medium-en"),
    device=device("cuda:0"),
)

completion = pipeline(
    prompt,
    max_length=1024,
    do_sample=True,
    top_k=40,
    temperature=1.0,
    top_p=0.9,
)

print(completion[0]["generated_text"])
```
Poetry can also be generated easily. For more involved usage examples
take a look at the provided [examples](examples).
```python
from torch import device
from transformers.pipelines.text_generation import TextGenerationPipeline

from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.utils import Poetry2Tokens

model = ByGPT5LMHeadModel.from_pretrained("nllg/poetry-bygpt5-base-en")
tokenizer = ByGPT5Tokenizer.from_pretrained("nllg/poetry-bygpt5-base-en")
p2t = Poetry2Tokens(tokenizer)

pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device("cuda:0"),
)

styles = (
    tokenizer.bos_token
    + p2t.rhymes2tokens["ABAB"]
    + p2t.meters2tokens["iambus"]
    + p2t.alliterations2tokens["medium"]
)

quatrain = pipeline(
    styles,
    return_full_text=False,
    bad_words_ids=[[id_] for id_ in tokenizer.additional_special_tokens_ids],
    do_sample=True,
    max_length=384,
    top_k=0,
    temperature=0.7,
    top_p=0.9,
)

print(quatrain[0]["generated_text"])
```

## Released Model Checkpoints
We have released the following checkpoints for pre-trained ByGPT5 language
models on the [Hugging Face Model Hub](https://huggingface.co/nllg):

| ByGPT5 | Parameters | Language Modeling | Poetry Generation |
|:-------|:-----------|:------------------|:------------------|
| Small  | 73.5M      | [English](https://huggingface.co/nllg/bygpt5-small-en), [German](https://huggingface.co/nllg/bygpt5-small-de) | [English](https://huggingface.co/nllg/poetry-bygpt5-small-en), [German](https://huggingface.co/nllg/poetry-bygpt5-small-de) |
| Base   | 139.2M     | [English](https://huggingface.co/nllg/bygpt5-base-en), [German](https://huggingface.co/nllg/bygpt5-base-de) | [English](https://huggingface.co/nllg/poetry-bygpt5-base-en), [German](https://huggingface.co/nllg/poetry-bygpt5-base-de) |
| Medium | 289.1M     | [English](https://huggingface.co/nllg/bygpt5-medium-en), [German](https://huggingface.co/nllg/bygpt5-medium-de) | [English](https://huggingface.co/nllg/poetry-bygpt5-medium-en), [German](https://huggingface.co/nllg/poetry-bygpt5-medium-de) |
