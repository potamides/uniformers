# Uniformers<br><sub><sup>Token-free Language Modeling with ByGPT5 & Friends</sup></sub>

[![ACL Anthology](https://img.shields.io/badge/View%20on%20ACL%20Anthology-B31B1B?labelColor=gray&color=f8f9fa&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4wIgogICB3aWR0aD0iNjgiCiAgIGhlaWdodD0iNDYiCiAgIGlkPSJzdmcyIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNCIgLz4KICA8cGF0aAogICAgIGQ9Ik0gNDEuOTc3NTUzLC0yLjg0MjE3MDllLTAxNCBDIDQxLjk3NzU1MywxLjc2MTc4IDQxLjk3NzU1MywxLjQ0MjExIDQxLjk3NzU1MywzLjAxNTggTCA3LjQ4NjkwNTQsMy4wMTU4IEwgMCwzLjAxNTggTCAwLDEwLjUwMDc5IEwgMCwzOC40Nzg2NyBMIDAsNDYgTCA3LjQ4NjkwNTQsNDYgTCA0OS41MDA4MDIsNDYgTCA1Ni45ODc3MDgsNDYgTCA2OCw0NiBMIDY4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDEwLjUwMDc5IEwgNTYuOTg3NzA4LDMuMDE1OCBDIDU2Ljk4NzcwOCwxLjQ0MjExIDU2Ljk4NzcwOCwxLjc2MTc4IDU2Ljk4NzcwOCwtMi44NDIxNzA5ZS0wMTQgTCA0MS45Nzc1NTMsLTIuODQyMTcwOWUtMDE0IHogTSAxNS4wMTAxNTUsMTcuOTg1NzggTCA0MS45Nzc1NTMsMTcuOTg1NzggTCA0MS45Nzc1NTMsMzAuOTkzNjggTCAxNS4wMTAxNTUsMzAuOTkzNjggTCAxNS4wMTAxNTUsMTcuOTg1NzggeiAiCiAgICAgc3R5bGU9ImZpbGw6I2VkMWMyNDtmaWxsLW9wYWNpdHk6MTtmaWxsLXJ1bGU6ZXZlbm9kZDtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MTIuODk1NDExNDk7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW1pdGVybGltaXQ6NDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLWRhc2hvZmZzZXQ6MDtzdHJva2Utb3BhY2l0eToxIgogICAgIGlkPSJyZWN0MjE3OCIgLz4KPC9zdmc+Cg==)](https://aclanthology.org/2023.acl-long.406)
[![arXiv]( https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2212.10474)
[![Semantic Scholar](https://img.shields.io/badge/View%20on%20Semantic%20Scholar-0f3875?logo=semanticscholar&labelColor=gray&logoColor=f4d35e)](https://www.semanticscholar.org/paper/ByGPT5%3A-End-to-End-Style-conditioned-Poetry-with-Belouadi-Eger/11ddb0953eae196dab339bfdc117221594cf945e)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ast-seDV6_pSzCvpGapNfTV_qDe0KRdF)

[Uniformers](https://github.com/potamides/uniformers) (short for _**Uni**versal
Coded Character Set Trans**formers**_) is a library for token-free language
modeling. In particular, it contains the reference implementation of _**ByGPT5:
End-to-End Style-conditioned Poetry Generation with Token-free Language
Models**_. ByGPT5 is a token-free decoder-only transformer that excels at
character-level tasks such as style-conditioned poetry generation. 

* :scroll: Read our [paper](https://arxiv.org/abs/2212.10474) on ByGPT5 for details.
* :feather: An interactive demo for poetry generation is [available](https://colab.research.google.com/drive/1Ast-seDV6_pSzCvpGapNfTV_qDe0KRdF).
* :bulb: If you make use of this library in your work please [cite](https://aclanthology.org/2023.acl-long.406.bib) it.

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

## Released Datasets
By default, this library creates QuaTrain on-the-fly when needed (which can
take some time). A preprocessed version (both in English and German) can be
found under [releases](https://github.com/potamides/uniformers/releases/latest).

| Dataset  | Language | #Quatrains |
|:---------|:---------|:-----------|
| QuaTrain | [English](https://github.com/potamides/uniformers/releases/latest/download/QuaTrain-en.json) | 2.7M |
| QuaTrain | [German](https://github.com/potamides/uniformers/releases/latest/download/QuaTrain-de.json)  | 5.9M |
