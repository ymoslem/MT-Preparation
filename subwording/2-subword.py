#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Subwording the source and target files
# Command: python3 subword.py <sp_source_model_path> <sp_target_model_path> <source_file_path> <target_file_path>


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

with open(source_raw, encoding='utf-8') as source, open(source_subworded, "w+", encoding='utf-8') as source_subword:
    for line in source:
        line = line.strip()
        line = sp.encode_as_pieces(line)
        # line = ['<s>'] + line + ['</s>']    # add start & end tokens; optional
        line = " ".join([token for token in line])
        source_subword.write(line + "\n")

print("Done subwording the source file! Output:", source_subworded)


# Subwording the train target

sp.load(target_model)

with open(target_raw, encoding='utf-8') as target, open(target_subworded, "w+", encoding='utf-8') as target_subword:
    for line in target:
        line = line.strip()
        line = sp.encode_as_pieces(line)
        # line = ['<s>'] + line + ['</s>']    # add start & end tokens; unrequired for OpenNMT
        line = " ".join([token for token in line])
        target_subword.write(line + "\n")

print("Done subwording the target file! Output:", target_subworded)

