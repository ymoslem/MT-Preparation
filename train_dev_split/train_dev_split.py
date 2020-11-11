#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Splitting the parallel dataset into train and development datasets for Machine Translation
# Command: python3 train_dev_split.py <dev_segment_number> <source_file_path> <target_file_path>


import pandas as pd
import numpy as np
from sacremoses import MosesTokenizer
import re
import csv
import sys

# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
# from IPython.display import display


segment_no = sys.argv[1]    # Number of segments in the dev set
source_file = sys.argv[2]   # Path to the source file
target_file = sys.argv[3]   # Path to the target file


def extract_dev(segment_no, source_file, target_file):
    
    df_source = pd.read_csv(source_file, names=['Source'], sep="\n", quoting=csv.QUOTE_NONE, error_bad_lines=False)
    df_target = pd.read_csv(target_file, names=['Target'], sep="\n", quoting=csv.QUOTE_NONE, error_bad_lines=False)
    df = pd.concat([df_source, df_target], axis=1)  # Join the two dataframes along columns
    print("Dataframe shape:", df.shape)
    
    
    # Delete rows with empty cells (source or target)
    df = df.dropna()
    
    print("--- Empty Cells Deleted", "--> Rows:", df.shape[0])
    
    
    # Extract Dev set from the main dataset
    df_dev = df.sample(n = int(segment_no))
    df_train = df.drop(df_dev.index)
    
    
    # Write the dataframe to two Source and Target files
    source_file_train = source_file+'.train'
    target_file_train = target_file+'.train'
    
    source_file_dev = source_file+'.dev'
    target_file_dev = target_file+'.dev'
    
    
    df_dic_train = df_train.to_dict(orient='list')
    
    
    with open(source_file_train, "w") as sf:
        sf.write("\n".join(line for line in df_dic_train['Source']))
    
    with open(target_file_train, "w") as tf:
        tf.write("\n".join(line for line in df_dic_train['Target']))
    
    
    df_dic_dev = df_dev.to_dict(orient='list')
    
    with open(source_file_dev, "w") as sf:
        sf.write("\n".join(line for line in df_dic_dev['Source']))
    
    with open(target_file_dev, "w") as tf:
        tf.write("\n".join(line for line in df_dic_dev['Target']))
        

    print("--- Wrote Files")
    print("Done!")
    print("Output files", *[source_file_train, target_file_train, source_file_dev, target_file_dev], sep="\n")
    


extract_dev(segment_no, source_file, target_file)
