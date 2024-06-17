# MT-Preparation
Machine Translation (MT) Preparation Scripts

## Installing Requirements

The filtering and subwording scripts use a number of Python packages. To install these dependencies using `pip` run the following command in Terminal/CMD:

```
pip3 install --user -r requirements.txt
```

## Filtering
There is one script to use for cleaning your Machine Translation dataset. You must have two files, one for the source and one for the target. If you rather have one TMX file, you can first use the [TXM2MT](https://github.com/ymoslem/file-converters) converter.

The filter script achieves the following steps:
* Deleting empty rows;
* Deleting duplicates;
* Deleting source-copied rows;
* Deleting too long Source/Target (ratio 200% and > 200 words);
* Removing HTML;
* Segments will remain in the true-case unless lower is True;
* Shuffling rows; and
* writing the output files.

Run the filtering script in the Terminal/CMD as follows:
```
python3 filter.py <source_file_path> <target_file_path> <source_lang> <target_lang>
```

## Subwording

It is recommended to run the subwording process, as it helps your Machine Translation engine avoid out-of-vocabulary tokens. The subwording scripts apply [SentencePiece](https://github.com/google/sentencepiece) to your source and target Machine Translation files. There are three scripts provided:

### 1. Train a subwording model

You need to create two subwording models to learn the vocabulary of your source and target.

```
python3 train.py <train_source_file_tok> <train_target_file_tok>
```

By default, the subwording model type is `unigram`. You can change it BPE by adding `--model_type=bpe` to these lines in the script as follows:

```
source_train_value = '--input='+train_source_file_tok+' --model_prefix=source --vocab_size='+str(source_vocab_size)+' --hard_vocab_limit=false --model_type=bpe'
```

```
target_train_value = '--input='+train_target_file_tok+' --model_prefix=target --vocab_size='+str(target_vocab_size)+' --hard_vocab_limit=false --model_type=bpe'
```

Optionally, you can add [more options](https://github.com/google/sentencepiece/blob/master/doc/options.md) like `--split_digits=true` to split all digits (0-9) into separate pieces, or `--byte_fallback=true` to decompose unknown pieces into UTF-8 byte pieces, which might help avoid out of vocabulary tokens. 

**Notes for big corpora:**

* You can use `--train_extremely_large_corpus=true` for a big corpus to avoid memory issues.
* The default SentencePiece value for `--input_sentence_size` is 0, i.e. the whole corpus. You can change it to a value between 1 and 10 million sentences, which will be enough for creating a good SentencePiece model.
* When the value of `--input_sentence_size` is less than the size of the corpus, it is recommended to set `--shuffle_input_sentence=true` to make your sample representative to the distribution of your data.
* The default SentencePiece value for `--vocab_size` is 8,000. You can go for a higher value between 30,000 and 50,000, and up to 100,000 for a big corpus. Still, note that smaller values will encourage the model to make more splits on words, which might be better in the case of a multilingual model if the languages share the alphabet.

### 2. Subword

In this step, you use the models you created in the previous step to subword your source and target Machine Translation files. You have to apply the same step on the source files to be translated later with the Machine Translation model.

```
python3 subword.py <sp_source_model_path> <sp_target_model_path> <source_file_path> <target_file_path>
```

**Notes for OpenNMT users:**

* If you are using OpenNMT, you can add `<s>` and `</s>` to the source only. Remove `<s>` and `</s>` from the target as they are already added by default ([reference](https://forum.opennmt.net/t/end-and-start-tokens/4570/2)). Alternatively, in OpenNMT-tf, there is an option called `source_sequence_controls` to add `start` and/or `end` tokens to the source.
* After you segment your source and target files with the generated SentencePiece models, you must [build vocab](https://opennmt.net/OpenNMT-py/options/build_vocab.html) using OpenNMT-py to generate vocab files compatible with it. OpenNMT-tf has an option that allows [converting SentencePiece vocab](https://opennmt.net/OpenNMT-tf/vocabulary.html#convert-a-sentencepiece-vocabulary-to-opennmt-tf) to a compatible format.
* Before you start training with OpenNMT-py, you must configure `src_vocab_size` and `tgt_vocab_size` to exactly match the value you used for `--vocab_size` in SentencePiece. The default is 50000, which is usually good.


### 3. Desubword

This step is useful after training your Machine Translation model and translating files with it, as you need to decode/desubword the generated target (i.e. translated) files.

```
python3 desubword.py <target_model_file> <target_pred_file>
```


## Extracting Training and Development Datasets

In this step, you split the parallel dataset into training and development datasets. The first argument is the number of segments you want in the development dataset; the script randomly selects this number of segments for the dev set and keeps the rest for the train set.

```
python3 train_dev_split.py <dev_segment_number> <source_file_path> <target_file_path>
```


## Google Colab Notebooks

* [Data Gathering and Processing](https://colab.research.google.com/drive/1rsFPnAQu9-_A6e2Aw9JYK3C8mXx9djsF)
* [Training Transformer-based NMT Model with OpenNMT-py](https://colab.research.google.com/drive/1HU8YKp52njmjtROjN-UNneTu0oYxYg9i)


## Questions
If you have questions or suggestions, please feel free to [contact](https://blog.machinetranslation.io/contact/) me.


## Citation

```bibtex
@inproceedings{moslem-etal-2022-domain,
    title = "Domain-Specific Text Generation for Machine Translation",
    author = "Moslem, Yasmin  and
      Haque, Rejwanul  and
      Kelleher, John  and
      Way, Andy",
    booktitle = "Proceedings of the 15th biennial conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)",
    month = sep,
    year = "2022",
    address = "Orlando, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2022.amta-research.2",
    pages = "14--30",
}
```

