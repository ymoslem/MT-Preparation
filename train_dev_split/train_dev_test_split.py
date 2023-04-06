#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Splitting the parallel dataset into train, development and test datasets for Machine Translation
# Command: python3 train_dev_split.py <dev_segment_number> <test_segment_number> <source_file_path> <target_file_path>


import pandas as pd
import numpy as np
import re
import csv
import sys

# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
# from IPython.display import display


segment_no_dev = sys.argv[1]    # Number of segments in the dev set
segment_no_test = sys.argv[2]    # Number of segments in the test set
source_file = sys.argv[3]   # Path to the source file
target_file = sys.argv[4]   # Path to the target file


def extract_dev(segment_no_dev, segment_no_test, source_file, target_file):

    df_source = pd.read_csv(source_file,
                            names=['Source'],
                            sep="\0",
                            quoting=csv.QUOTE_NONE,
                            skip_blank_lines=False,
                            on_bad_lines="skip")
    df_target = pd.read_csv(target_file,
                            names=['Target'],
                            sep="\0",
                            quoting=csv.QUOTE_NONE,
                            skip_blank_lines=False,
                            on_bad_lines="skip")
    df = pd.concat([df_source, df_target], axis=1)  # Join the two dataframes along columns
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
    source_file_train = source_file+'.train'
    target_file_train = target_file+'.train'

    source_file_dev = source_file+'.dev'
    target_file_dev = target_file+'.dev'

    source_file_test = source_file+'.test'
    target_file_test = target_file+'.test'

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



extract_dev(segment_no_dev, segment_no_test, source_file, target_file)
