#!/bin/bash 

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src

####################################################
################### LANGUAGE MODEL ################
####################################################

##### varying training data size and representation unit ###

dataset=news12

outdir=$MLXLIT_BASE/results/lm_eval/1_train_size/$dataset


#mkdir -p $outdir
#for lang in `echo hi kn` #bn ta`
#do 
#    for exp in `echo phonetic onehot` 
#    do 
#        rm $outdir/${exp}_${lang}.txt
#        rm $outdir/${exp}_${lang}.log
#    
#        #for train_size in `echo "1000 2000 5000 15000 25000 35000"`
#        for train_size in `echo "1000 2000 5000 8000 10000 13000"`
#        do 
#            echo "Running Training size: $train_size"
#            python $MLXLIT_HOME/src/lm_eval/ptb_word_lm.py \
#                --data_path=/home/development/anoop/experiments/multilingual_unsup_xlit/data/lm_eval/$dataset/$lang \
#                --representation=$exp \
#                --train_size=$train_size \
#                --lang=$lang >> \
#                $outdir/${exp}_${lang}.txt 2>> \
#                $outdir/${exp}_${lang}.log
#        done 
#    done 
#done
#

for lang in `echo hi kn`
do 
    for exp in `echo phonetic onehot` 
    do 
        cat $outdir/${exp}_${lang}.txt | grep -i 'Test Perplexity' | sed 's,Test Perplexity:,,g' | sed  "s,^,$exp $lang,g"
    done 
done 

