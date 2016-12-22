#!/bin/bash 

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=/home/development/anoop/experiments/unsupervised_transliterator/src/transliterator
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 

export CUDA_VISIBLE_DEVICES=0

#####################################################
#################### LANGUAGE MODEL ################
#####################################################
#
###### varying training data size and representation unit ###
#
#dataset=news12
#
#outdir=$MLXLIT_BASE/results/lm_eval/1_train_size/$dataset
#
#
##mkdir -p $outdir
##for lang in `echo hi kn` #bn ta`
##do 
##    for exp in `echo phonetic onehot` 
##    do 
##        rm $outdir/${exp}_${lang}.txt
##        rm $outdir/${exp}_${lang}.log
##    
##        #for train_size in `echo "1000 2000 5000 15000 25000 35000"`
##        for train_size in `echo "1000 2000 5000 8000 10000 13000"`
##        do 
##            echo "Running Training size: $train_size"
##            python $MLXLIT_HOME/src/lm_eval/ptb_word_lm.py \
##                --data_path=/home/development/anoop/experiments/multilingual_unsup_xlit/data/lm_eval/$dataset/$lang \
##                --representation=$exp \
##                --train_size=$train_size \
##                --lang=$lang >> \
##                $outdir/${exp}_${lang}.txt 2>> \
##                $outdir/${exp}_${lang}.log
##        done 
##    done 
##done
##
#
#for lang in `echo hi kn`
#do 
#    for exp in `echo phonetic onehot` 
#    do 
#        cat $outdir/${exp}_${lang}.txt | grep -i 'Test Perplexity' | sed 's,Test Perplexity:,,g' | sed  "s,^,$exp $lang,g"
#    done 
#done 
#

################# Unsupervised transliteration #########################
#
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/unsup/conll16
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/unsup/conll16
#
#expname='1_newrep'
#
#for langpair in `echo hi-kn bn-hi ta-kn`
##for langpair in `echo bn-hi ta-kn`
#do
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    for representation in `echo onehot phonetic`
#    do 
#        o=$output_dir/$expname/$representation/$langpair
#        
#        mkdir -p $o
#
#        echo 'Start: ' $expname $langpair $representation 
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            --train_mode unsup \
#            --langs "$src_lang,$tgt_lang" \
#            --data_dir  $data_dir/$langpair \
#            --output_dir  $o \
#            --representation $representation > $o/train.log 2>&1 
#
#
#        echo 'End: ' $expname $langpair $representation 
#
#    done 
#done     

################# Semi supervised transliteration #########################
#
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/semisup/conll16
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/semisup/conll16
#
#expname='1_newrep'
#
#for langpair in `echo hi-kn bn-hi ta-kn`
##for langpair in `echo bn-hi ta-kn`
#do
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    for representation in `echo onehot phonetic`
#    do 
#        o=$output_dir/$expname/$representation/$langpair
#        
#        mkdir -p $o
#
#        echo 'Start: ' $expname $langpair $representation 
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            --train_mode semisup \
#            --lang_pairs "$src_lang-$tgt_lang" \
#            --data_dir  $data_dir/$langpair \
#            --output_dir  $o \
#            --representation $representation > $o/train.log 2>&1 
#
#        echo 'End: ' $expname $langpair $representation 
#
#    done 
#done     


################# supervised transliteration #########################

data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/conll16
## refdir contains same ref as data_dir, but in XML format required for evaluation tools. These are from the CoNLL 2016 directories
ref_dir=~/experiments/unsupervised_transliterator/data/nonparallel/pb
output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/conll16

#expname='5_sup_nomono'
##train_bidirectional='--train_bidirectional'
##use_monolingual='--use_monolingual'

#expname='2_bisup_nomono_again'
#train_bidirectional='--train_bidirectional'
##use_monolingual='--use_monolingual'

#expname='3_sup_mono'
##train_bidirectional='--train_bidirectional'
#use_monolingual='--use_monolingual'

#expname='4_bisup_mono'
#train_bidirectional='--train_bidirectional'
#use_monolingual='--use_monolingual'

### XXXXXXX NOTE: remove temporary option for use of single language for monolingual 

#for expname in `echo 1_sup_nomono 2_bisup_nomono 3_sup_mono 4_bisup_mono `
for expname in `echo 3_3_use_src`
do 

    exptype=`echo $expname | cut -f 1 -d '_'`

    ######### choose specific model  ############

    if [ $exptype = '1' ]  # sup_nomono
    then 
        train_bidirectional=''
        use_monolingual=''
    elif [ $exptype = '2' ] # bisup_nomono 
    then 
        train_bidirectional='--train_bidirectional'
        use_monolingual=''
    elif [ $exptype = '3' ] # sup_mono 
    then 
        train_bidirectional=''
        use_monolingual='--use_monolingual --which_mono 0'
    elif [ $exptype = '4' ] # bisup_mono
    then 
        train_bidirectional='--train_bidirectional'
        use_monolingual='--use_monolingual'
    fi 
   
    ######## Experiment loop starts here ########

    #for langpair in `echo hi-kn bn-hi ta-kn`
    for langpair in `echo bn-hi ta-kn`
    do
        src_lang=`echo $langpair | cut -f 1 -d '-'`
        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
    
        for representation in `echo onehot phonetic`
        do 
            o=$output_dir/$expname/$representation/$langpair
            
            echo 'Start: ' $expname $langpair $representation 
    
            ## Training and Testing 
            rm -rf $o
            mkdir -p $o
            python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
                --train_mode sup \
                $train_bidirectional \
                $use_monolingual \
                --lang_pairs "$src_lang-$tgt_lang" \
                --data_dir  $data_dir/$langpair \
                --output_dir  $o \
                --representation $representation > $o/train.log 2>&1 
    
            ### Evaluation starts 
    
            prefix=`ls $o/outputs/ | sed 's,[^0-9],,g' | sort -r -n | head -1`
    
            ## convert to required format 
            python utilities.py convert_output_format  \
                $o/outputs/${prefix}${src_lang}-${tgt_lang}_ \
                $o/outputs/${prefix}test.${tgt_lang} 
    
            # convert to n-best format 
            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py  convert_to_nbest_format  \
                $o/outputs/${prefix}test.${tgt_lang}  $o/outputs/${prefix}test.nbest.${tgt_lang}
            
            # generate NEWS 2015 evaluation format output file 
            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
                    "$ref_dir/$src_lang-$tgt_lang/test.id" \
                    "$ref_dir/$src_lang-$tgt_lang/test.xml" \
                    "$o/outputs/${prefix}test.nbest.${tgt_lang}" \
                    "$o/outputs/${prefix}test.nbest.${tgt_lang}.xml" \
                    "system" "conll2016" "$src_lang" "$tgt_lang"  
            
            # run evaluation 
            python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
                    -t "$ref_dir/$src_lang-$tgt_lang/test.xml" \
                    -i "$o/outputs/${prefix}test.nbest.${tgt_lang}.xml" \
                    -o "$o/outputs/${prefix}test.nbest.${tgt_lang}.detaileval.csv" \
                    > "$o/outputs/${prefix}test.nbest.${tgt_lang}.eval"
    
            echo 'End: ' $expname $langpair $representation 
    
        done 
    done     


done 
