# MT-Preparation
Machine Translation (MT) Preparation Scripts

## Installing Requirements

The filtering and subwording scripts use a number of Python packages. To install these dependencies using `pip` run the following command in Terminal/CMD:

```
pip3 install --user -r requirements.txt
```

## Filtering
There is one script to use for clearning your Machine Translation dataset. You must have two files, one for the source and one for the target. If you rather have one TMX file, you can first use the [TXM2MT](https://github.com/ymoslem/file-converters) converter.

The filter script achieve the following steps:
* deleting empty cells;
* removing HTML tags;
* tokenizing the Source segments;
* tokenizing the Target segments
* deleting duplicates;
* deleting source-copied rows;
* deleting too-long Source/Target (ratio 200%); and
* writing the output files.

Run the filtering script in the Terminal/CMD as follows:
```
python3 filter.py <source_lang> <target_lang> <source_file_path> <target_file_path>
```

## Subwording

It is recommended to run the subwording process as it helps your Machine Translation engine avoid out-of-vocabulary tokens. The subwording scripts apply [SentencePiece](https://github.com/google/sentencepiece) on your source and target Machine Translation files. There are three scripts provided:

**1. Train a subwording model**

You need to create two subwording models to learn the vocabulary of your source and target.

```
python3 train.py <train_source_file_tok> <train_target_file_tok>
```

By default, the subwording model type is `unigram`. You can change it BPE by adding `--model_type=bpe` to this line in the script as follows:

```
source_train_value = '--input='+train_source_file_tok+' --model_prefix=source --vocab_size='+str(source_vocab_size)+' --hard_vocab_limit=false --model_type=bpe'
```

```
target_train_value = '--input='+train_target_file_tok+' --model_prefix=target --vocab_size='+str(target_vocab_size)+' --hard_vocab_limit=false --model_type=bpe'
```

**2. Subword**

In this step, you use the models you created in the previous step to subword your source and target Machine Translation files. You have to apply the same step on the source files to be translated later with the Machine Translation model.

```
python3 subword.py <source_model_file> <target_model_file> <source_pred_file> <target_pred_file>
```

**3. Desubword**

This step is useful after training your Machine Translation model and translating files with it as you need to decode/desubword the generated target (i.e. translated) files.

```
python3 desubword.py <target_model_file> <target_pred_file>
```

## Extracting Training and Development Datasets

In this step, you split the parallel dataset into training and development datasets. The first argument is the number of segments you want in the development dataset; the script rondomly selects this number of segments for the dev set and keeps the rest for the train set.

```
python3 train_dev_split.py <dev_segment_number> <source_file_path> <target_file_path>
```

## Questions
If you have questions or suggestions, please feel free to [contact](https://blog.machinetranslation.io/contact/) me.

