#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Splitting the parallel dataset into train, development and test datasets for Machine Translation
# Command: python3 train_dev_split.py <dev_segment_number> <test_segment_number> <bilingual_file_path> <source_lang> <target_lang>


import pandas as pd
import numpy as np
import re
import csv
import sys

# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
# from IPython.display import display


segment_no_dev = sys.argv[1]    # Number of segments in the dev set
segment_no_test = sys.argv[2]    # Number of segments in the test set
bilingual_file = sys.argv[3]   # Path to the bilingual file
source_lang = sys.argv[4]    # source language
target_lang = sys.argv[5]    # target language


def extract_dev(segment_no_dev, segment_no_test, bilingual_file, source_lang, target_lang):

    df = pd.read_csv(bilingual_file, names=['Source', 'Target'], sep="\t", quoting=csv.QUOTE_NONE, skip_blank_lines=False, on_bad_lines="skip")
    print("Dataframe shape:", df.shape)


    # Delete rows with empty cells (source or target)
    df = df.dropna()

    print("--- Empty Cells Deleted", "--> Rows:", df.shape[0])


    # Extract Dev set from the main dataset
    df_dev = df.sample(n = int(segment_no_dev))
    df_train = df.drop(df_dev.index)

    # Extract Test set from the main dataset
    df_test = df_train.sample(n = int(segment_no_test))
    df_train = df_train.drop(df_test.index)

    # Write the dataframe to two Source and Target files
    source_file_train = f"{bilingual_file}.{source_lang}.train"
    target_file_train = f"{bilingual_file}.{target_lang}.train"

    source_file_dev = f"{bilingual_file}.{source_lang}.dev"
    target_file_dev = f"{bilingual_file}.{target_lang}.dev"

    source_file_test = f"{bilingual_file}.{source_lang}.test"
    target_file_test = f"{bilingual_file}.{target_lang}.test"

    df_dic_train = df_train.to_dict(orient='list')


    with open(source_file_train, "w") as sf:
        sf.write("\n".join(line for line in df_dic_train['Source']))
        sf.write("\n") # end of file newline

    with open(target_file_train, "w") as tf:
        tf.write("\n".join(line for line in df_dic_train['Target']))
        tf.write("\n") # end of file newline


    df_dic_dev = df_dev.to_dict(orient='list')

    with open(source_file_dev, "w") as sf:
        sf.write("\n".join(line for line in df_dic_dev['Source']))
        sf.write("\n") # end of file newline
        
    with open(target_file_dev, "w") as tf:
        tf.write("\n".join(line for line in df_dic_dev['Target']))
        tf.write("\n") # end of file newline
        

    df_dic_test = df_test.to_dict(orient='list')

    with open(source_file_test, "w") as sf:
        sf.write("\n".join(line for line in df_dic_test['Source']))
        sf.write("\n") # end of file newline
        
    with open(target_file_test, "w") as tf:
        tf.write("\n".join(line for line in df_dic_test['Target']))
        tf.write("\n") # end of file newline
        

    print("--- Wrote Files")
    print("Done!")
    print("Output files", *[source_file_train, target_file_train, source_file_dev, target_file_dev, source_file_test, target_file_test], sep="\n")



extract_dev(segment_no_dev, segment_no_test, bilingual_file, source_lang, target_lang)
