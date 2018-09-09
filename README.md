# MLXLIT

_A Multilingual Neural Machine Transliteration System_

A multilingual neural machine translation system written in TensorFlow. Although it has been primarily tested for transliteration, it could be used for translation also. It implements an attention-based neural MT system. For Indian scripts, it implements a few specific features:
  - phonetic encoding for Indic scripts 
  - mapping characters in different Indian scripts into a common script

## Pre-requisites

Python packages required:

- Python 2.7x
- Tensorflow 0.11
- mpld3
- sklearn
- matplotlib
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)
- [IIT Bombay Unsupervised transliteration](https://github.com/anoopkunchukuttan/transliterator)(_required for computing evaluation metrics for transliteration, and for analysis or visualization_)

## Training 


### Training Dataset Format

The data files are all in the format: one sequence per line, separated by spaces. They are all organized in a directory which has the following structure. It contains the following sub-directories: 

`parallel_train`: Containing training data. Contains two files for every language pair in training: `<src_lang>-<tgt_lang>.<src_lang>` (source file) and `<src_lang>-<tgt_lang>.<tgt_lang>` (target file). 
`parallel_valid`: Containing validation data. Contains two files for every language pair in training: `<src_lang>-<tgt_lang>.<src_lang>` (source file) and `<src_lang>-<tgt_lang>.<tgt_lang>` (target file). 
`test`: Containing test data.  Contains two files for every language pair in training: `<src_lang>-<tgt_lang>` (source file) and `<tgt_lang>-<src_lang>` (target file). In addition, the directory contains two more files for every language pair for evaluation: 
  - `test.<src_lang>-<tgt_lang>.xml`: xml file in the format required by the NEWS shared task evaluation scripts.
  - `test.<src_lang>-<tgt_lang>.id`: A text file with one line for every sequence in the dataset. Each line contains the following text: `<seqno>_1_0`.  `seqno` starts from 1 e.g. `10_1_0`

### Training a Model 

To train the models, use the `src/unsup_xlit/ModelTraining.py` script. A common training run is as follows: 

```bash 

python ModelTraining.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--lang_pairs LANG_PAIRS]

optional arguments:
  --data_dir DATA_DIR   data directory (default: None)
  --output_dir OUTPUT_DIR
                        output folder name (default: None)
  --lang_pairs LANG_PAIRS
                        List of language pairs for supervised training given
                        as: "lang1-lang2,lang3-lang4,..." (default: None)
```


The following are all the commandline options available for running the training script: 

```bash 

usage: ModelTraining.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
                        [--train_size N] [--lang_pairs LANG_PAIRS]
                        [--unseen_langs UNSEEN_LANGS] [--enc_type ENC_TYPE]
                        [--separate_output_embedding] [--prefix_tgtlang]
                        [--prefix_srclang] [--embedding_size EMBEDDING_SIZE]
                        [--enc_rnn_size ENC_RNN_SIZE]
                        [--dec_rnn_size DEC_RNN_SIZE]
                        [--max_seq_length MAX_SEQ_LENGTH]
                        [--batch_size BATCH_SIZE] [--max_epochs MAX_EPOCHS]
                        [--learning_rate LEARNING_RATE]
                        [--dropout_keep_prob DROPOUT_KEEP_PROB]
                        [--infer_every INFER_EVERY] [--topn TOPN]
                        [--beam_size BEAM_SIZE] [--start_from START_FROM]
                        [--representation REPRESENTATION]
                        [--shared_mapping_class SHARED_MAPPING_CLASS]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   data directory (default: None)
  --output_dir OUTPUT_DIR
                        output folder name (default: None)
  --train_size N        use the first N sequence pairs for training. N=-1
                        means use the entire training set. (default: -1)
  --lang_pairs LANG_PAIRS
                        List of language pairs for supervised training given
                        as: "lang1-lang2,lang3-lang4,..." (default: None)
  --unseen_langs UNSEEN_LANGS
                        List of languages not seen during training given as:
                        "lang1,lang2,lang3,lang4,..." (default: None)
  --enc_type ENC_TYPE   encoder to use. One of (1) simple_lstm_noattn (2)
                        bilstm (3) cnn (default: cnn)
  --separate_output_embedding
                        Should separate embeddings be used on the input and
                        output side. Generally the same embeddings are to be
                        used. This is used only for Indic-Indic
                        transliteration, when input is phonetic and output is
                        onehot_shared (default: False)
  --prefix_tgtlang      Prefix the input sequence with the language code for
                        the target language (default: False)
  --prefix_srclang      Prefix the input sequence with the language code for
                        the source language (default: False)
  --embedding_size EMBEDDING_SIZE
                        size of character representation (default: 256)
  --enc_rnn_size ENC_RNN_SIZE
                        size of output of encoder RNN (default: 512)
  --dec_rnn_size DEC_RNN_SIZE
                        size of output of dec RNN (default: 512)
  --max_seq_length MAX_SEQ_LENGTH
                        maximum sequence length (default: 30)
  --batch_size BATCH_SIZE
                        size of each batch used in training (default: 32)
  --max_epochs MAX_EPOCHS
                        maximum number of epochs (default: 30)
  --learning_rate LEARNING_RATE
                        learning rate of Adam Optimizer (default: 0.001)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        keep probablity for the dropout layers (default: 0.5)
  --infer_every INFER_EVERY
                        write predicted outputs for test data after these many
                        epochs, 0 if not required (default: 1)
  --topn TOPN           the number of best candidates to output by the decoder
                        (default: 10)
  --beam_size BEAM_SIZE
                        beam size for decoding (default: 5)
  --start_from START_FROM
                        epoch to restore model from. This must be one of the
                        epochs for which model has been saved (default: None)
  --representation REPRESENTATION
                        input representation, which can be specified in two
                        ways: (i) one of "phonetic", "onehot",
                        "onehot_and_phonetic" - this representation is used
                        for all languages, (ii) "lang1:<rep1>,lang2:<rep2>" -
                        this representation is used for all languages, rep can
                        be one of "phonetic", "onehot", "onehot_and_phonetic"
                        (default: onehot)
  --shared_mapping_class SHARED_MAPPING_CLASS
                        class to be used for shared mapping. Possible values:
                        IndicPhoneticMapping, CharacterMapping (default:
                        IndicPhoneticMapping)

```


The output directory has the following structure: 

`train.log`: log file generated during training 
`mappings`: directory containing vocabulary of all the languages and vocabulary to id mappings. A JSON file for every languages' vocabulary is found in this directory with the name `mapping_<lang>.json`. e.g. `my_model-1` 
`temp_models`: direcory containing saved models. The saved models are named as `my_model-<epoch_number>`. `<epoch_number>` is not zero padded. e.g. `my_model-1`
`outputs`: Directory containing output after decoding test set with models saved. The output file is named as `<epoch_number>test.nbest.<src_lang>-<tgt_lang>.<tgt_lang>. <epoch_number>` is 3-digit zero-padded. e.g. `001test.nbest.en-hi.hi`
`validation`: Directory containing output after decoding test set with models saved. The output file is named as `<epoch_number>test.nbest.<src_lang>-<tgt_lang>.<tgt_lang>`. `<epoch_number>` is 3-digit zero-padded. e.g. `001test.nbest.en-hi.hi`

`final_output`: ignore this directory

## Decoding 

To decode using the trained models, use the `src/unsup_xlit/ModelDecoding.py` script. A common training run is as follows: 

```

python ModelDecoding.py [--max_seq_length MAX_SEQ_LENGTH]
                        [--batch_size BATCH_SIZE] [--enc_type ENC_TYPE]
                        [--separate_output_embedding] [--prefix_tgtlang]
                        [--prefix_srclang] [--embedding_size EMBEDDING_SIZE]
                        [--enc_rnn_size ENC_RNN_SIZE]
                        [--dec_rnn_size DEC_RNN_SIZE]
                        [--representation REPRESENTATION]
                        [--shared_mapping_class SHARED_MAPPING_CLASS]
                        [--topn TOPN] [--beam_size BEAM_SIZE]
                        [--lang_pair LANG_PAIR] [--model_fname MODEL_FNAME]
                        [--mapping_dir MAPPING_DIR] [--in_fname IN_FNAME]
                        [--out_fname OUT_FNAME]

optional arguments:
   --max_seq_length MAX_SEQ_LENGTH
                        maximum sequence length (default: 30)
  --batch_size BATCH_SIZE
                        size of each batch used in decoding (default: 100)
  --enc_type ENC_TYPE   encoder to use. One of (1) simple_lstm_noattn (2)
                        bilstm (3) cnn (default: cnn)
  --separate_output_embedding
                        Should separate embeddings be used on the input and
                        output side. Generally the same embeddings are to be
                        used. This is used only for Indic-Indic
                        transliteration, when input is phonetic and output is
                        onehot_shared (default: False)
  --prefix_tgtlang      Prefix the input sequence with the language code for
                        the target language (default: False)
  --prefix_srclang      Prefix the input sequence with the language code for
                        the source language (default: False)
  --embedding_size EMBEDDING_SIZE
                        size of character representation (default: 256)
  --enc_rnn_size ENC_RNN_SIZE
                        size of output of encoder RNN (default: 512)
  --dec_rnn_size DEC_RNN_SIZE
                        size of output of dec RNN (default: 512)
  --representation REPRESENTATION
                        input representation, which can be specified in two
                        ways: (i) one of "phonetic", "onehot",
                        "onehot_and_phonetic" (default: onehot)
  --shared_mapping_class SHARED_MAPPING_CLASS
                        class to be used for shared mapping. Possible values:
                        IndicPhoneticMapping, CharacterMapping (default:
                        IndicPhoneticMapping)
  --topn TOPN           The top-n candidates to report (default: 10)
  --beam_size BEAM_SIZE
                        beam size for decoding (default: 5)
  --lang_pair LANG_PAIR
                        language pair for decoding: "lang1-lang2" (default:
                        None)
  --model_fname MODEL_FNAME
                        model file name (default: None)
  --mapping_dir MAPPING_DIR
                        directory containing mapping files (default: None)
  --in_fname IN_FNAME   input file (default: None)
  --out_fname OUT_FNAME
                        results file (default: None)

```

For parameters related to the network architecture, use the same parameters used for training the network. 


## Citing this work

If you use this code in any of your work, please cite: 
   
Anoop Kunchukuttan, Mitesh Khapra, Gurneet Singh, Pushpak Bhattacharyya. _Leveraging Orthographic Similarity for Multilingual Neural Transliteration_. Transactions of Association of Computational Linguistics. 2018. 


## Comparing with our work

The code is written in Tensorflow 0.11 and has not been migrated to the recent version of Tensorflow. We are no longer maintaining this codebase. There are two ways to reproduce our work: 

- Compile Tensorflow 0.11 from source and then run our system 
- For transliteration with one-hot representations, you could use any off-the-shelf NMT system and share encoders/decoders across all the languages. You could also use the special language token trick used in Google's multilingual Neural MT system to support multiple target languages. As shown in our paper, this approach is quite competitive with our architecture (where we have a target language specific output layer).

## Authors

- Anoop Kunchukuttan
- Gurneet Singh
- Mitesh Khapra 

## Contact

Anoop Kunchukuttan: <anoop.kunchukuttan@gmail.com> or Prof. Pushpak Bhattacharyya: <pb@cse.iitb.ac.in>

## License

Copyright Anoop Kunchukuttan 2016 - present
 
MLXLIT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MLXLIT is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with MLXLIT.  If not, see <http://www.gnu.org/licenses/>.



