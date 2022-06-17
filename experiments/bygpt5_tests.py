#!/usr/bin/env python

import torch
from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer

tokenizer = ByGPT5Tokenizer.from_pretrained("google/byt5-small")
model = ByGPT5LMHeadModel.from_pretrained("google/byt5-small")

def generate(sent):
    input_ids = torch.tensor([list(sent.encode("utf-8"))]) + 3
    greedy_output = model.generate(input_ids, min_length=50, max_length=50, do_sample=True) # pyright: ignore
    print([tokenizer._convert_id_to_token(id_) for id_ in greedy_output[0]])
    print("<", tokenizer.decode(greedy_output[0]), ">")


generate("Wonderfu")
