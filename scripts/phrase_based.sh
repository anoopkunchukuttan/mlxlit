#!/bin/bash 

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=$MLXLIT_BASE/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 

###Building phrase-based systems


######   make LM  ###
#dataset=slavic_latin-ar 
#lm_dir=$MLXLIT_BASE/results/pbsmt/$dataset/lm/
#data_dir=$MLXLIT_BASE/data/sup/mosesformat/$dataset/
#
#mkdir -p $lm_dir

#for lang in `echo cs pl sk sl`
#do 
#    ngram-count -wbdiscount -interpolate \
#        -text $data_dir/ar-$lang/train.$lang \
#        -lm $lm_dir/$lang.5g.lm \
#        -order 5 
#done 
    

###  train models

#parallel --joblog 'ar-slavic.job.log' --jobs 2 --gnu "nohup time /usr/local/bin/smt/moses_job_scripts/moses_run.sh run_params.ar-{}.conf > ar-{}.log 2>&1" <<  LANG_PAIRS
#pl
#cs
#sk
#sl
#LANG_PAIRS

#parallel --joblog 'slavic-ar.job.log' --jobs 2 --gnu "nohup time /usr/local/bin/smt/moses_job_scripts/moses_run.sh run_params.{}-ar.conf > {}-ar.log 2>&1" <<  LANG_PAIRS
#pl
#cs
#sk
#sl
#LANG_PAIRS

### evaluate 

dataset=slavic_latin-ar
data_dir=$MLXLIT_BASE/data/sup/mosesformat/$dataset
o=$MLXLIT_BASE/results/pbsmt/$dataset

for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
do 
    src_lang=`echo $langpair | cut -f 1 -d '-'`
    tgt_lang=`echo $langpair | cut -f 2 -d '-'`

    # generate NEWS 2015 evaluation format output file 
    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
            "$data_dir/$langpair/test.id" \
            "$data_dir/$langpair/test.xml" \
            "$o/$langpair/evaluation/test.nbest.${tgt_lang}" \
            "$o/$langpair/evaluation/test.nbest.${tgt_lang}.xml" \
            "system" "news2015" "$src_lang" "$tgt_lang"  
    
    # run evaluation 
    python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
            -t "$data_dir/$langpair/test.xml" \
            -i "$o/$langpair/evaluation/test.nbest.${tgt_lang}.xml" \
            -o "$o/$langpair/evaluation/test.nbest.${tgt_lang}.detaileval.csv" \
             > "$o/$langpair/evaluation/test.nbest.${tgt_lang}.eval"

done 


######
###  dir=news_2015_indic  ; find $dir -name '*.eval' | xargs head -1q | cut -f 2 -d ':' | tr -d ' ' ; find $dir -name '*.eval' | cut -f 2 -d '/'
######


