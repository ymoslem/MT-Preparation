#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filtering/Cleaning parallel datasets for Machine Translation
# Please read the steps and adjust them for your needs.
# Command: python3 filter.py <source_file_path> <target_file_path> <source_lang> <target_lang>


import pandas as pd
import numpy as np
import re
import csv
import sys

# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
#from IPython.display import display


def prepare(source_file, target_file, source_lang, target_lang, lower=False):
    
    df_source = pd.read_csv(source_file, names=['Source'], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False, on_bad_lines="skip")
    df_target = pd.read_csv(target_file, names=['Target'], sep="\0", quoting=csv.QUOTE_NONE, skip_blank_lines=False, on_bad_lines="skip")
    df = pd.concat([df_source, df_target], axis=1)  # Join the two dataframes along columns
    print("Dataframe shape (rows, columns):", df.shape)

    
    # Delete nan
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])


    # Drop duplicates
    df = df.drop_duplicates()
    #df = df.drop_duplicates(subset=['Target'])

    print("--- Duplicates Deleted\t\t\t--> Rows:", df.shape[0])


    # Drop copy-source rows
    df["Source-Copied"] = df['Source'] == df['Target']
    #display(df.loc[df['Source-Copied'] == True]) # display only copy-sourced rows
    df = df.set_index(['Source-Copied'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no source-copied cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass
    
    df = df.reset_index()
    df = df.drop(['Source-Copied'], axis = 1)
    
    print("--- Source-Copied Rows Deleted\t\t--> Rows:", df.shape[0])


    # Drop too-long rows (source or target)
    # Based on your language, change the values "2" and "200"
    df["Too-Long"] = ((df['Source'].str.count(' ')+1) > (df['Target'].str.count(' ')+1) * 2) |  \
                     ((df['Target'].str.count(' ')+1) > (df['Source'].str.count(' ')+1) * 2) |  \
                     ((df['Source'].str.count(' ')+1) > 200) |  \
                     ((df['Target'].str.count(' ')+1) > 200)
                
    #display(df.loc[df['Too-Long'] == True]) # display only too long rows
    df = df.set_index(['Too-Long'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass

    df = df.reset_index()
    df = df.drop(['Too-Long'], axis = 1)

    print("--- Too Long Source/Target Deleted\t--> Rows:", df.shape[0])


    # Remove HTML and normalize
    # Use str() to avoid (TypeError: expected string or bytes-like object)
    # Note: removing tags should be before removing empty cells because some cells might have only tags and become empty.

    df = df.replace(r'<.*?>|&lt;.*?&gt;|&?(amp|nbsp|quot);|{}', ' ', regex=True)
    df = df.replace(r'  ', ' ', regex=True)  # replace double-spaces with one space

    print("--- HTML Removed\t\t\t--> Rows:", df.shape[0])


    # Lower-case the data
    if lower == True:
        df['Source'] = df['Source'].str.lower()
        df['Target'] = df['Target'].str.lower()

        print("--- Rows are now lower-cased\t--> Rows:", df.shape[0])
    else:
        print("--- Rows will remain in true-cased\t--> Rows:", df.shape[0])


    # Replace empty cells with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Delete nan (already there, or generated from the previous steps)
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])


    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    print("--- Rows Shuffled\t\t\t--> Rows:", df.shape[0])


    # Write the dataframe to two Source and Target files
    source_file = source_file+'-filtered.'+source_lang
    target_file = target_file+'-filtered.'+target_lang


    # Save source and target to two text files
    df_source = df["Source"]
    df_target = df["Target"]

    df_source.to_csv(source_file, header=False, index=False, quoting=csv.QUOTE_NONE, sep="\n")
    print("--- Source Saved:", source_file)
    df_target.to_csv(target_file, header=False, index=False, quoting=csv.QUOTE_NONE, sep="\n")
    print("--- Target Saved:", target_file)


# Corpora details
source_file = sys.argv[1]    # path to the source file
target_file = sys.argv[2]    # path to the target file
source_lang = sys.argv[3]    # source language
target_lang = sys.argv[4]    # target language

# Run the prepare() function
# Data will be true-case; change to True to lower-case
prepare(source_file, target_file, source_lang, target_lang, lower=False)
