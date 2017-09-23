# MLXLIT

_A Multilingual Neural Machine Transliteration System_

## Pre-requisites

Python packages required

- Tensorflow 0.11
- mpld3
- sklearn
- matplotlib
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)
- [IIT Bombay Unsupervised transliteration](https://github.com/anoopkunchukuttan/transliterator)(_required for computing evaluation metrics for transliteration, and for analysis or visualization_)

## Training a Model


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

For details on more parameters, run `python ModelTraining.py --help`

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

For details on more parameters, run `python ModelDecoding.py --help`. For parameters related to the network architecture, use the same parameters used for training the network. 


## Authors

- Anoop Kunchukuttan
- Gurneet Singh
- Mitesh Khapra 

## Revision Log 

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



