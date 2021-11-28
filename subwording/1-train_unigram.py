#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Training SentencePiece models for the source and target
# Command: python3 train.py <train_source_file_tok> <train_target_file_tok>


import sys
import sentencepiece as spm


path = ""    # change the path if needed

train_source_file_tok = path + sys.argv[1]
train_target_file_tok = path + sys.argv[2]
    

# train sentencepiece model from the source and target files
# and create `source/target.model` and `source/target.vocab`
# `source/target.vocab` is just a reference, not used in the segmentation.

# if the training data is too small and the maximum pieces reserved is less than 4000.
# decrease --vocab size or --hard_vocab_limit=false, which automatically shrink the vocab size.


# Source subword model

source_train_value = '--input='+train_source_file_tok+' --model_prefix=source --vocab_size=50000 --hard_vocab_limit=false --split_digits=true'
spm.SentencePieceTrainer.train(source_train_value)
print("Done, training a SentencepPiece model for the Source finished successfully!")


# Target subword model

target_train_value = '--input='+train_target_file_tok+' --model_prefix=target --vocab_size=50000 --hard_vocab_limit=false --split_digits=true'
spm.SentencePieceTrainer.train(target_train_value)
print("Done, training a SentencepPiece model for the Target finished successfully!")

