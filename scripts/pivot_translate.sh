#!/bin/bash 

MOSES_HOME="/usr/local/bin/smt/mosesdecoder-latest-25Dec2015"
MOSES_CMD="$MOSES_HOME/bin/moses"
SCRIPTS_ROOTDIR="$MOSES_HOME/scripts"

pivot_method=$1

model_base_dir=$2
outdir=$3

exp=$4
src_lang=$5
pivot_lang=$6
tgt_lang=$7

data_dir=$8  # parallel corpus directory, needed for tm-triangulation tuning only
repunit=$9 # needed for tm-triangulation tuning only

shift 9

sp_dir=$model_base_dir/$exp/$src_lang-$pivot_lang
pt_dir=$model_base_dir/$exp/$pivot_lang-$tgt_lang 
spt_dir=$outdir/$exp/$src_lang-$pivot_lang-$tgt_lang 

## Path to TM triangulate installation 
TMT_DIR=/home/development/anoop/installs/MultiMT

## create the directory to store the final results 
mkdir -p $spt_dir/{input,moses_data,evaluation,log,tuning}

echo '### Pivoting for experiment and language triple: ' $exp $src_lang $pivot_lang $tgt_lang 
echo '### Method:' $pivot_method 

######## TRANSFER BASED PIVOT METHOD ##########

### decode src into pivot   (or copy it if available) 
output_1=$spt_dir/input/test.nbest.$pivot_lang
input_1st_fname=$data_dir/$repunit/$src_lang-$pivot_lang/test.$src_lang
#cp $sp_dir/evaluation/test.nbest.$pivot_lang $output_1

python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
    --lang_pair $langpair \
    --beam_size 5 \
    --mapping_dir "$o/mappings" \
    --model_fname "$o/temp_models/my_model-$prefix1"  \
    --representation $rep_str \
    --in_fname    "$data_dir/$langpair/test/$langpair" \
    --out_fname   "/home/development/anoop/tmp/test.$tgt_lang"
    #--out_fname   "$o/outputs/${prefix}test.nbest.$langpair.$tgt_lang"

echo 'Stage 1 done for experiment and language triple: ' $exp $src_lang $pivot_lang $tgt_lang 

### get the input for next stage 
input_2nd_fname=$spt_dir/input/input_second_stage.$pivot_lang
sed  's/ ||| /|/g;s/ |/|/g' $output_1 | \
    cut -d'|' -f2  > $input_2nd_fname

echo 'Prepared Input for Stage 2 for experiment and language triple: ' $exp $src_lang $pivot_lang $tgt_lang 

#### decode brige into target
### number of threads and distortion limit picked from from tuned.ini, so not explicitly specified
output_2=$spt_dir/input/test.nbest.$pivot_lang-$tgt_lang
$MOSES_CMD -f $pt_dir/tuning/moses.ini \
    -n-best-list $output_2 20 \
    -alignment-output-file "$spt_dir/evaluation/test.align.$pivot_lang-$tgt_lang" \
    -output-unknowns "$spt_dir/evaluation/test.oov.$pivot_lang-$tgt_lang" \
    $MOSES_DECODER_OPTS_PT \
    < $input_2nd_fname > $spt_dir/input/test.$pivot_lang-$tgt_lang 2> $spt_dir/log/test.log 

echo 'Stage 2 translation done for experiment and language triple: ' $exp $src_lang $pivot_lang $tgt_lang 

### creating the final output file 
final_output=$spt_dir/evaluation/test.nbest.$tgt_lang
python utilities.py transfer_pivot_translate $output_1 $output_2 $final_output

### generate one best output from n-best list 
python utilities.py convert_to_1best_format $final_output $spt_dir/evaluation/test.$tgt_lang

echo 'Combined Stage 1 and Stage 2 results for experiment and language triple: ' $exp $src_lang $pivot_lang $tgt_lang 

