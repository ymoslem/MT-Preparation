#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Filtering/Cleaning parallel datasets for Machine Translation
# Command: python3 filter.py <source_file_path> <target_file_path>


import pandas as pd
import numpy as np
import re
import csv
import sys

# display(df) works only if you are in IPython/Jupyter Notebooks or enable:
# from IPython.display import display


def prepare(source_file, target_file, lower=False):

    df_source = pd.read_csv(source_file, names=['Source'], sep="\n", quoting=csv.QUOTE_NONE)
    df_target = pd.read_csv(target_file, names=['Target'], sep="\n", quoting=csv.QUOTE_NONE)
    df = pd.concat([df_source, df_target], axis=1)  # Join the two dataframes along columns
    print("Dataframe shape (rows, columns):", df.shape)
    
    
    # Lower-case the data
    if lower == True:
        df['Source'] = df['Source'].str.lower()
        df['Target'] = df['Target'].str.lower()

    # Delete nan
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])



    # Tokenize and lower-case text, and remove HTML.
    # Use str() to avoid (TypeError: expected string or bytes-like object)
    # Note: removing tags should be before removing empty cells because some cells might have only tags and become empty.

    html = re.compile('<.*?>|&lt;.*?&gt;') # Maybe also &quot;

    clean_source = lambda row: re.sub(html, '', str(row)).strip()

    df["Source"] = df["Source"].apply(clean_source)

    print("--- Cleaning the Source Complete\t--> Rows:", df.shape[0])


    clean_target = lambda row: re.sub(html, '', str(row)).strip()

    df["Target"] = df["Target"].apply(clean_target)

    print("--- Cleaning the Target Complete\t--> Rows:", df.shape[0])
    

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
    df["Too-Long"] = (df['Source'].str.len() > df['Target'].str.len() * 1.5) | (df['Target'].str.len() > df['Source'].str.len() * 1.5)
    #display(df.loc[df['Too long'] == True]) # display only too long rows
    df = df.set_index(['Too-Long'])

    try: # To avoid (KeyError: '[True] not found in axis') if there are no too-long cells
        df = df.drop([True]) # Boolean, not string, do not add quotes
    except:
        pass
    
    df = df.reset_index()
    df = df.drop(['Too-Long'], axis = 1)

    print("--- Too-Long Source/Target Deleted\t--> Rows:", df.shape[0])


    # Replace empty cells with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Delete nan (already there, or generated from the previous steps)
    df = df.dropna()

    print("--- Rows with Empty Cells Deleted\t--> Rows:", df.shape[0])
    
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    print("--- Rows Shuffled\t\t\t--> Rows:", df.shape[0])
    
    
    # Write the dataframe to two Source and Target files
    source_file = source_file+'-filtered'
    target_file = target_file+'-filtered'


    df_dic = df.to_dict(orient='list')

    with open(source_file, "w") as sf:
        sf.write("\n".join(line for line in df_dic['Source']))
        sf.write("\n")

    with open(target_file, "w") as tf:
        tf.write("\n".join(line for line in df_dic['Target']))
        tf.write("\n")

    print("--- Wrote Files")
    print("Done!")
    print("Output files:\n• ", source_file, "\n• ", target_file)


# Corpora details
source_file = sys.argv[1]    # path to the source file
target_file = sys.argv[2]    # path to the target file
prepare(source_file, target_file, lower=False)
