#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Training SentencePiece models for the source and target jointly.
# First, merge the source and target files, and maybe shuffle (shuf) too.
# cat <train_source_file> <train_target_file> > <train_joint_file>
# Then, train a joint SentencePiece model, that covers both the source and target
# python3 train.py <train_joint_file>


import sys
import sentencepiece as spm


train_joint_file = sys.argv[1]

# Train a subword model
spm.SentencePieceTrainer.train(input=train_joint_file,
                               model_prefix="aren",
                               model_type="unigram",
                               vocab_size=32000,
                               input_sentence_size=20000000,
                               shuffle_input_sentence=True,
                               train_extremely_large_corpus=True,
                               split_digits=True,
                               byte_fallback=True,
                               user_defined_symbols=["&apos;", "&quot;", "&amp;", "&lt;", "&gt;",
                                                     "<s>", "</s>", "<2ar>", "<2en>", "<BT>",
                                                     "<LEGAL>", "<NEWS>", "<HEALTH>",
                                                     "<t0>", "<t1>", "<t2>", "<t3>", "<t4>", "<t5>",
                                                 ]
                            )


print("Done, training the SentencePiece model finished successfully!")