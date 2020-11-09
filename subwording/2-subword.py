#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Subwording the source and target files
# Command: python3 subword.py <source_model_file> <target_model_file> <source_pred_file> <target_pred_file>


import sys
import sentencepiece as spm


source_model = sys.argv[1]
target_model = sys.argv[2]
source_raw = sys.argv[3]
target_raw = sys.argv[4]
source_subworded = source_raw + ".subword"
target_subworded = target_raw + ".subword"

print("Source Model:", source_model)
print("Target Model:", target_model)
print("Source Dataset:", source_raw)
print("Target Dataset:", target_raw)


sp = spm.SentencePieceProcessor()


# Subwording the train source

sp.load(source_model)

with open(source_raw) as source, open(source_subworded, "w+") as source_subword:
    for line in source:
        line = ['<s>'] + sp.encode_as_pieces(line) + ['</s>']    # encode and add start & end tokens
        line = " ".join([token for token in line])
        source_subword.write(line + "\n")

print("Done subwording the source file! Output:", source_subworded)


# Subwording the train target

sp.load(target_model)

with open(target_raw) as target, open(target_subworded, "w+") as target_subword:
    for line in target:
        line = ['<s>'] + sp.encode_as_pieces(line) + ['</s>']    # encode and add start & end tokens
        line = " ".join([token for token in line])
        target_subword.write(line + "\n")

print("Done subwording the target file! Output:", target_subworded)

