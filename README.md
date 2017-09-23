# Multilingual Neural Machine Transliteration System 

## Pre-requisites

Python packages required

- Tensorflow 0.11
- mpld3
- sklearn
- matplotlib
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)
- [IIT Bombay Unsupervised transliteration](https://github.com/anoopkunchukuttan/transliterator)(_required only if you are interested in analysis or visualization_)

## Training a Model


### Training Dataset Format 

### Training a Model 

To train the models, use the `src/unsup_xlit/ModelTraining.py` script. A common training run is as follows: 

```bash 

python ModelTraining.py [-h] [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
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
  --train_size N        use the first N sequence pairs for training. N=-1 means use the entire training set. (default:
                        -1)
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
  --topn TOPN           The top-n candidates to output by the decoder (default: 10)
  --beam_size BEAM_SIZE
                        beam size for decoding (default: 5)
  --start_from START_FROM
                        epoch to restore model from. This must be one of the epochs for which model has been saved (default: None)
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

## Decoding 

## Adding a New Language


## Authors

- Anoop Kunchukuttan
- Gurneet Singh
- Mitesh Khapra 

## Revision Log 

## License

Copyright Anoop Kunchukuttan 2016 - present
 
YYYYY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

YYYYY is distributed in the hope that it will be useful, 
but WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with Indic NLP Library.  If not, see <http://www.gnu.org/licenses/>.



