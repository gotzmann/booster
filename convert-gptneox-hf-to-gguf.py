#!/usr/bin/env python3
# HF gptneox--> gguf conversion

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer  # type: ignore[import]

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py' / 'gguf'))
import gguf


def count_model_parts(dir_model: Path) -> int:
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("pytorch_model-"):
            num_parts += 1

    if num_parts > 0:
        print("gguf: found " + str(num_parts) + " model parts")
    return num_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a GPT-NeoX model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file, or model file itself (*.bin)",
    )
    parser.add_argument(
        "ftype", type=int, choices=[0, 1], default=1, nargs='?',
        help="output format - use 0 for float32, 1 for float16",
    )
    return parser.parse_args()

args = parse_args()

dir_model = args.model
ftype = args.ftype
if not dir_model.is_dir():
    print(f'Error: {args.model} is not a directory', file = sys.stderr)
    sys.exit(1)

# possible tensor data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16

# map from ftype to string
ftype_str = ["f32", "f16"]

if args.outfile is not None:
    fname_out = args.outfile
else:
    # output in the same directory as the model by default
    fname_out = dir_model / f'ggml-model-{ftype_str[ftype]}.gguf'

print("gguf: loading model "+dir_model.name)

with open(dir_model / "config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

if hparams["architectures"][0] != "GPTNeoXForCausalLM":
    print("Model architecture not supported: " + hparams["architectures"][0])

    sys.exit()

# get number of model parts
num_parts = count_model_parts(dir_model)

ARCH=gguf.MODEL_ARCH.GPTNEOX
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["num_hidden_layers"]

gguf_writer.add_name(dir_model.name)
gguf_writer.add_context_length(hparams["max_position_embeddings"])
gguf_writer.add_embedding_length(hparams["hidden_size"])
gguf_writer.add_block_count(block_count)
gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
gguf_writer.add_rope_dimension_count(int(hparams["rotary_pct"]*(hparams["hidden_size"]//hparams["num_attention_heads"])))
gguf_writer.add_head_count(hparams["num_attention_heads"])
gguf_writer.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
gguf_writer.add_layer_norm_eps(hparams["layer_norm_eps"])

# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: list[bytearray] = []
scores: list[float] = []
toktypes: list[int] = []

# gpt2 tokenizer
gguf_writer.add_tokenizer_model("gpt2")

print("gguf: get gpt2 tokenizer vocab")

# ref: https://github.com/cmp-nct/ggllm.cpp/blob/master/falcon_convert.py
tokenizer = AutoTokenizer.from_pretrained(dir_model)

# The number of tokens in tokenizer.json can differ from the expected vocab size.
# This causes downstream issues with mismatched tensor sizes when running the inference
vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
assert max(tokenizer.vocab.values()) < vocab_size

added_vocab = tokenizer.get_added_vocab()
reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}

for i in range(vocab_size):
    if i not in reverse_vocab:
        tokens.append(f"[PAD{i}]")
        toktypes.append(gguf.TokenType.USER_DEFINED)
    elif reverse_vocab[i] in added_vocab:
        tokens.append(reverse_vocab[i])
        if tokenizer.added_tokens_decoder[i].special:
            toktypes.append(gguf.TokenType.CONTROL)
        else:
            toktypes.append(gguf.TokenType.USER_DEFINED)
    else:
        tokens.append(reverse_vocab[i])
        toktypes.append(gguf.TokenType.NORMAL)

gguf_writer.add_token_list(tokens)
gguf_writer.add_token_types(toktypes)

special_vocab = gguf.SpecialVocab(dir_model, load_merges = True, n_vocab = len(tokens))
special_vocab.add_to_gguf(gguf_writer)

# TENSORS

tensor_map = gguf.get_tensor_name_map(ARCH,block_count)

# tensor info
print("gguf: get tensor metadata")

if num_parts == 0:
    part_names = iter(("pytorch_model.bin",))
else:
    part_names = (
        f"pytorch_model-{n:05}-of-{num_parts:05}.bin" for n in range(1, num_parts + 1)
    )

for part_name in part_names:
    if args.vocab_only:
        break
    print("gguf: loading model part '" + part_name + "'")
    model_part = torch.load(f"{dir_model}/{part_name}", map_location="cpu")

    for name in model_part.keys():
        data = model_part[name]

        # we don't need these
        if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
            continue

        old_dtype = data.dtype

        # convert any unsupported data types to float32
        if data.dtype != torch.float16 and data.dtype != torch.float32:
            data = data.to(torch.float32)

        data = data.squeeze().numpy()

        # map tensor names
        new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
        if new_name is None:
            print("Can not map tensor '" + name + "'")
            sys.exit()

        n_dims = len(data.shape)
        data_dtype = data.dtype

        # if f32 desired, convert any float16 to float32
        if ftype == 0 and data_dtype == np.float16:
            data = data.astype(np.float32)

        # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
        if ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
            data = data.astype(np.float16)

        print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

        gguf_writer.add_tensor(new_name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
if not args.vocab_only:
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

gguf_writer.close()

print(f"gguf: model successfully exported to '{fname_out}'")
print("")
