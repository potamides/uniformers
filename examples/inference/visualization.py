#!/usr/bin/env python
from argparse import ArgumentParser
from html import escape
from re import escape as regesc, sub
from textwrap import dedent

from torch import cat, mean, norm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM as AutoLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from uniformers.models.bygpt5 import ByGPT5LMHeadModel, ByGPT5Tokenizer
from uniformers.utils import Poetry2Tokens
from uniformers.vendor.alti import (
    ModelWrapper,
    compute_joint_attention,
    normalize_contributions,
)

# https://stackoverflow.com/a/25875504
def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
        '\n': r'\\',
    }
    regex = '|'.join(regesc(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item)))
    return sub(regex, lambda m: conv[m.group()], text)

class Visualizer:
    colormap = {
        0.0: "FFFFFF",
        0.2: "ADCCF6",
        0.4: "79ABE2",
        0.6: "2C86CA",
        0.8: "005D9A",
        1.0: "00366C",
    }

    def __init__(self, model, tokenizer, sequence, alti=True, rescale=True, keep_eos_bos=False):
        assert isinstance(model, (ByGPT5LMHeadModel, GPT2LMHeadModel))
        inputs = tokenizer(sequence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

        special_start, special_end = tokens[:4], [tokens[-1]]
        sequence = list(tokenizer.convert_tokens_to_string(tokens[4:-1]))

        assert all(
            token in tokenizer.all_special_tokens
            for token in special_start + special_end
        )
        assert all(token not in tokenizer.all_special_tokens for token in sequence)

        self.rescale = rescale
        self.keep_eos_bos = keep_eos_bos
        self.tokenizer = tokenizer
        self.tokens = special_start + sequence + special_end
        if alti:
            self.attentions = self._alti(inputs, model)
        else:
            outputs = model(**inputs)  # pyright: ignore
            self.attentions = mean(cat(outputs[-1]), dim=1)

        # map attentions to bytes
        if isinstance(model, GPT2LMHeadModel):
            indeces = [0, 1, 2, 3] # special tokens at the begining
            for idx, token in enumerate(tokens[4:-1], len(indeces)):
                indeces.extend([idx] * len(token))
            indeces.append(len(tokens) - 1) # eos
            self.attentions = self.attentions[:, :, indeces]

    def _alti(self, inputs, model):
        model_wrapped = ModelWrapper(model)
        *_, contributions_data = model_wrapped(inputs)
        resultant_norm = norm(
            contributions_data["resultants"].squeeze(), p=1, dim=-1  # pyright: ignore
        )
        normalized_contributions = normalize_contributions(
            contributions_data["contributions"],
            scaling="min_sum",
            resultant_norm=resultant_norm,
        )
        contributions_mix = compute_joint_attention(normalized_contributions)

        return contributions_mix

    def _rescale(self, attention):
        max_ = attention.max()
        min_ = attention.min()
        rescale = (attention - min_) / (max_ - min_)
        return rescale

    def _tokens2rgb(self, indeces, layer):
        attention = mean(self.attentions[layer][indeces], dim=0)
        attention = self._rescale(attention) if self.rescale else attention
        colors = list()

        offset = 0
        for tok in self.tokens:
            if tok.strip():
                if tok in self.tokenizer.all_special_tokens:
                    att_len = 1
                else:
                    # handle characters which occupy multiple utf8 bytes
                    att_len = len(tok.encode("utf-8"))
                att = attention[offset : offset + att_len].mean()
                offset += att_len
                for threshold, color in self.colormap.items():
                    if att.round(decimals=1) <= threshold:
                        rgb = color
                        break
                else:
                    raise ValueError("Attention has illegal value!")
                colors.append(rgb)
            else:
                colors.append(self.colormap[0.0])
                offset += 1

        assert offset == len(attention)

        return colors

    def get_fg(self, color):
        red, green, blue = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        # https://stackoverflow.com/a/3943023
        if (red * 0.299 + green * 0.587 + blue * 0.114) > 186:
            return "black"
        return "white"

    def html(self, indeces, layer=-1):
        tokens, colors, html = self.tokens, self._tokens2rgb(indeces, layer), ""

        # heuristic: interpret zero attentions at end of sequence as not yet
        # generated tokens for highlighting. Not perfect but works
        for index in range(len(colors)):
            if all(color == self.colormap[0.0] for color in colors[index:]):
                position = index
                break
        else:
            raise ValueError

        if not self.keep_eos_bos:
            tokens = tokens[1:-1]
            colors = colors[1:-1]
            position -= 1

        for index, (token, color) in enumerate(zip(tokens, colors)):
            if token.strip():
                fg = self.get_fg(color)
                font = "monospace" if token in self.tokenizer.all_special_tokens else "serif"
                weight="bold" if index >= position else "normal"
                style = f"color:{fg};background-color:#{color};font-family:{font};font-weight:{weight}"
                if token in self.tokenizer.additional_special_tokens:
                    token = f"<{Poetry2Tokens(self.tokenizer).tokens2forms[token]}>"
                html += (
                    f'<span style="{style}">{escape(token)}</span>'
                )
            else:
                html += escape(token).replace("\n", "<br />")

        return html

    def tex(self, indeces, layer=-1):
        tokens, colors, verses = self.tokens, self._tokens2rgb(indeces, layer), ""
        special, t2f = list(), Poetry2Tokens(self.tokenizer).tokens2forms
        special_style = r"{{\texttt{{<{}>}}}}"
        color = r"\colorbox[HTML]{{{bg}}}{{\textcolor{{{fg}}}{{\strut{{}}{text}}}}}"
        tex = dedent(
            r"""
            \ifx\versesize\undefined\def\versesize{{\normalsize}}\fi%
            \ifx\stylesize\undefined\def\stylesize{{\small}}\fi%
            {{\versesize\fboxsep0pt{{}}
              \begin{{tabular}}{{cl}}
                \multirow{{4}}{{*}}{{\stylesize\makecell{{{special}}}}}
                & {verse1}\\
                & {verse2}\\
                & {verse3}\\
                & {verse4}
              \end{{tabular}}
            }}%
            \global\let\stylesize\undefined%
            \global\let\versesize\undefined%
            """
        )

        # heuristic: interpret zero attentions at end of sequence as not yet
        # generated tokens for highlighting. Not perfect but works
        for index in range(len(colors)):
            if all(color == self.colormap[0.0] for color in colors[index:]):
                position = index
                break
        else:
            raise ValueError

        if not self.keep_eos_bos:
            tokens = tokens[1:-1]
            colors = colors[1:-1]
            position -= 1

        for index, (token, bg) in enumerate(zip(tokens, colors)):
            if token.strip():
                fg = self.get_fg(bg)
                style = special_style if token in self.tokenizer.all_special_tokens else r"\textrm{{\textit{{{}}}}}"
                weight=r"\textbf{{{}}}" if index >= position else "{}"

                if token in self.tokenizer.additional_special_tokens:
                    token = t2f[token]
                    special.append(color.format(fg=fg, bg=bg, text=style.format(weight.format(tex_escape(token)))))
                else:
                    verses += color.format(fg=fg, bg=bg, text=style.format(weight.format(tex_escape(token))))
            else:
                verses += tex_escape(token)
        verses = verses.split(r"\\")

        return tex.format(special=r"\\".join(special), verse1=verses[0], verse2=verses[1], verse3=verses[2], verse4=verses[3])


if __name__ == "__main__":
    sample_sequence = "</s><extra_id_18><extra_id_7><extra_id_1>When I consider how my light is spent,\nEre half my days, in this dark world and wide,\nAnd that one Talent which is death to hide\nLodged with me useless, though my Soul more bent</s>"

    parser = ArgumentParser(
        description="Visualize input attribution for a given generated sequence. Supports HTML and TeX."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="nllg/poetry-bygpt5-base-en",
        help="name of the model in huggingface hub or path if local",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="do not use ALTI and only use raw attention values",
    )
    parser.add_argument(
        "--sequence",
        default=sample_sequence,
        help="the sequence including special tokens to visualize",
    )
    parser.add_argument(
        "--indeces",
        nargs="+",
        type=int,
        default=[-6],
        help="the indeces of the tokens to compute relevances for (averaged if multiple), must be set together with '--sequence'",
    )
    parser.add_argument(
        "--output",
        help="path to the output file (do not print to stdout)",
    )
    args = parser.parse_args()

    try:
        model = AutoLM.from_pretrained(
            args.model_name_or_path, output_attentions=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except (EnvironmentError, KeyError, ValueError):
        model = ByGPT5LMHeadModel.from_pretrained(
            args.model_name_or_path, output_attentions=True
        )
        tokenizer = ByGPT5Tokenizer.from_pretrained(args.model_name_or_path)

    visualizer = Visualizer(model, tokenizer, args.sequence, alti=not args.raw)

    if args.output:
        with open(args.output, "wb") as f:
            if args.output.endswith(".html"):
                f.write(visualizer.html(indeces=args.indeces).encode())
            elif args.output.endswith(".tex"):
                f.write(visualizer.tex(indeces=args.indeces).encode())
            else:
                raise ValueError("Filetype not supported!")
    else:
        print(visualizer.html(indeces=args.indeces))
