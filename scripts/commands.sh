#!/bin/bash

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=$MLXLIT_BASE/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 

export CUDA_VISIBLE_DEVICES=1

###############################################################################################
############################ supervised transliteration - multilingual #########################
################################################################################################

dataset='news_2015'
data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset

#restore_epoch_number="17"

for expname in `echo 2_multilingual`
do 

    ######## Experiment loop starts here ########

    #for representation in `echo onehot phonetic`
    for representation in `echo onehot_shared`
    do 

        if [ $dataset = 'news_2015' ]
        then 
            # for EN-INDIC

            ### normal run
            src_lang='en'
            tgt_langs=(hi bn kn ta)
            lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
            representations_param=`for x in ${tgt_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`

            #### for zeroshot training
            #src_lang='en'
            #tgt_langs=(bn kn ta)
            #all_langs=(hi bn kn ta)
            #lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`

            ## common  block
            multiconf=en-indic
            rep_str="en:onehot,$representations_param"

        elif [ $dataset = 'news_2015_reversed' ]
        then 
            ## for INDIC-EN

            ### normal run
            tgt_lang='en'
            src_langs=(hi bn kn ta)
            lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
            representations_param=`for x in ${src_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`

            #### for zeroshot training
            #tgt_lang='en'
            #src_langs=(bn kn ta)
            #all_langs=(hi bn kn ta)
            #lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`

            ## common  block
            multiconf=indic-en
            rep_str="en:onehot,$representations_param"

        elif [ $dataset = 'news_2015_indic' ]
        then
            ## for INDIC-INDIC 

            ## all pairs 
            #lang_pairs="bn-hi,bn-kn,bn-ta,hi-bn,hi-kn,hi-ta,kn-bn,kn-hi,kn-ta,ta-bn,ta-hi,ta-kn"
        
            #some language pairs: then separate run for zeroshot is not required 
            lang_pairs="bn-hi,bn-kn,hi-bn,hi-ta,kn-bn,kn-ta,ta-hi,ta-kn"

            ## common block 
            multiconf=indic-indic
            rep_str="$representation"
            if [ $representation = 'phonetic' ]
            then 
                more_opts="--separate_output_embedding"
            fi 

        else
            echo 'Invalid dataset' 
            exit 1
        fi 

        o=$output_dir/$expname/$representation/$multiconf
        
        echo 'Start: ' $dataset $expname $multiconf $representation 
    
        ### Training and Testing 
        rm -rf $o
        mkdir -p $o

        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
            --lang_pairs "$lang_pairs" \
            --data_dir  $data_dir/$multiconf \
            --output_dir  $o \
            --representation "$rep_str" \
             $more_opts >> $o/train.log 2>&1 
    
            #--start_from $restore_epoch_number \
    
        echo 'End: ' $dataset $expname $langpair $representation 
    
    done 

done 

#############################################################################################
######################## supervised transliteration - bilingual  ############################
#############################################################################################

#dataset='news_2015_reversed'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
##restore_epoch_number="17"
#
#for expname in `echo 2_bilingual`
#do 
#
#    ######## Experiment loop starts here ########
#
#    #for langpair in `echo hi-kn bn-hi ta-kn`
#    #for langpair in `echo hi-en bn-en kn-en ta-en`
#    #for langpair `echo bn-hi bn-kn bn-ta hi-bn hi-kn hi-ta kn-bn kn-hi kn-ta ta-bn ta-hi ta-kn`
#    for langpair in `echo hi-en`
#    do
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#        #for representation in `echo onehot phonetic`
#        for representation in `echo phonetic`
#        do 
#            o=$output_dir/$expname/$representation/$langpair
#            
#            echo 'Start: ' $dataset $expname $langpair $representation 
#
#            if [ $dataset = 'news_2015' ]
#            then 
#                rep_str="en:onehot,$tgt_lang:$representation"
#            elif [ $dataset = 'news_2015_reversed' ]
#            then 
#                rep_str="en:onehot,$src_lang:$representation"
#            elif [ $dataset = 'news_2015_indic' ]
#            then 
#                rep_str="$representation"
#                if [ $representation = 'phonetic' ]
#                then 
#                    more_opts="--separate_output_embedding"
#                fi 
#            else
#                echo 'Invalid dataset' 
#                exit 1
#            fi 
#    
#            ### Training and Testing 
#            rm -rf $o
#            mkdir -p $o
#
#            python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#                --lang_pairs "$src_lang-$tgt_lang" \
#                --data_dir   "$data_dir/$langpair" \
#                --output_dir "$o" \
#                --representation "$rep_str" \
#                $more_opts \
#                >> $o/train.log 2>&1 
#    
#                #--start_from $restore_epoch_number \
#
#            echo 'End: ' $dataset $expname $langpair $representation 
#    
#        done 
#    done     
#
#done 

##############################################################################################
###################### EVALUATE SPECIFIC ITERATION #################
##############################################################################################

#dataset='news_2015'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#for prefix in `echo  020 026`
#do 
#
#for expname in `echo 100_test_refactor`
#do 
#
#    ######## Experiment loop starts here ########
#
#    #for langpair in `echo en-hi en-bn en-ta en-kn`
#    #for langpair in `echo hi-en bn-en ta-en kn-en`
#    #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
#    for langpair in `echo en-bn`
#    do
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#        #for representation in `echo onehot phonetic`
#        for representation in `echo onehot`
#        do 
#            #### output directory to select 
#            ### for bilingual experiments 
#            o=$output_dir/$expname/$representation/$langpair
#
#            ### for multilingual experiments  (en-indic)
#            #o=$output_dir/$expname/$representation/en-indic
#
#            #### for multilingual experiments  (indic-en)
#            #o=$output_dir/$expname/$representation/indic-en
#            
#            #### for multilingual experiments  (indic-indic)
#            #o=$output_dir/$expname/$representation/indic-indic
#
#            echo 'Start: ' $dataset $expname $langpair $representation 
#    
#            #### Evaluation starts 
#            
#            resdir=outputs
#    
#            # generate NEWS 2015 evaluation format output file 
#            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                    "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.id" \
#                    "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
#                    "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#                    "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                    "system" "conll2016" "$src_lang" "$tgt_lang"  
#            
#            # run evaluation 
#            python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                    -t "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
#                    -i "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                    -o "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#                     > "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#    
#            echo 'End: ' $dataset $expname $langpair $representation 
#    
#        done 
#    done     
#done 
#done

#dataset='news_2015_reversed'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#while read e_r_l_p_v
#do 
#        expname=`echo "$e_r_l_p_v" | cut -f 1 -d '|'`
#        representation=`echo "$e_r_l_p_v" | cut -f 2 -d '|'`
#        langpair=`echo "$e_r_l_p_v" | cut -f 3 -d '|'`
#        prefix=`echo "$e_r_l_p_v" | cut -f 4 -d '|'`
#        loss=`echo "$e_r_l_p_v" | cut -f 5 -d '|'`
#
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#        echo $e_r_l_p_v
#        echo 
#
#        #### output directory to select 
#        #### for bilingual experiments 
#        o=$output_dir/$expname/$representation/$langpair
#
#        #### for multilingual experiments  (en-indic)
#        #o=$output_dir/$expname/$representation/en-indic
#
#        #### for multilingual experiments  (indic-en)
#        #o=$output_dir/$expname/$representation/indic-en
#        
#        ##### for multilingual experiments  (indic-indic)
#        #o=$output_dir/$expname/$representation/indic-indic
#
#        echo 'Start: ' $dataset $expname $langpair $representation 
#    
#        #### Evaluation starts 
#        
#        resdir=outputs
#    
#        # generate NEWS 2015 evaluation format output file 
#        python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.id" \
#                "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
#                "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#                "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                "system" "conll2016" "$src_lang" "$tgt_lang"  
#        
#        # run evaluation 
#        python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                -t "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
#                -i "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                -o "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#                 > "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#    
#        echo 'End: ' $dataset $expname $langpair $representation 
#done  <<CONFIG
#2_bilingual|onehot|hi-en|009|0.421373397112
#2_bilingual|onehot|bn-en|009|0.467538326979
#2_bilingual|onehot|kn-en|006|0.427557706833
#CONFIG

#1_multilingual_shared_decoder|onehot_shared|kn-hi|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|hi-kn|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|bn-ta|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|ta-bn|014|5.27229315042

########################################################################
################# Decoding #########################
########################################################################

##### for multilingual zeroshot training  (en-indic) with hindi as the missing language
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#while read e_r_l_p_v
#do 
#        expname=`echo "$e_r_l_p_v" | cut -f 1 -d '|'`
#        representation=`echo "$e_r_l_p_v" | cut -f 2 -d '|'`
#        langpair=`echo "$e_r_l_p_v" | cut -f 3 -d '|'`
#        prefix=`echo "$e_r_l_p_v" | cut -f 4 -d '|'`
#        loss=`echo "$e_r_l_p_v" | cut -f 5 -d '|'`
#
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#        prefix1=`echo $prefix | sed 's,^0\+,,g'`
#
#        echo $e_r_l_p_v
#        echo 
#
#        #### output directory to select 
#
#        if [ $dataset = 'news_2015' ]
#        then 
#            ### for multilingual experiments  (en-indic)
#            o=$output_dir/$expname/$representation/en-indic
#            rep_str="en:onehot,$tgt_lang:$representation"
#        elif [ $dataset = 'news_2015_reversed' ]
#        then 
#            ### for multilingual experiments  (indic-en)
#            o=$output_dir/$expname/$representation/indic-en
#            rep_str="en:onehot,$src_lang:$representation"
#        elif [ $dataset = 'news_2015_indic' ]
#        then 
#            ##### for multilingual experiments  (indic-indic)
#            o=$output_dir/$expname/$representation/indic-indic
#            rep_str="$representation" 
#            if [ $representation = 'phonetic' ]
#            then 
#                more_opts="--separate_output_embedding"
#            fi 
#        else
#            echo 'Invalid dataset'
#            exit 1
#        fi 
#
#        echo 'Start: ' $dataset $expname $langpair $representation 
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#            --lang_pair $langpair \
#            --beam_size 5 \
#            --mapping_dir "$o/mappings" \
#            --model_fname "$o/temp_models/my_model-$prefix1"  \
#            --representation $rep_str \
#            --in_fname    "$data_dir/$langpair/test/$langpair" \
#            --out_fname   "/home/development/anoop/tmp/test.$tgt_lang"
#            #--out_fname   "$o/outputs/${prefix}test.nbest.$langpair.$tgt_lang"
#
#        echo 'End: ' $dataset $expname $langpair $representation 
#
#done  <<CONFIG
#1_multilingual_shared_decoder|onehot_shared|kn-hi|014|5.27229315042
#CONFIG

### indic-indic
#1_multilingual_shared_decoder|onehot_shared|kn-hi|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|hi-kn|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|bn-ta|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|ta-bn|014|5.27229315042
#1_multilingual_shared_decoder|phonetic|kn-hi|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|hi-kn|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|bn-ta|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|ta-bn|014|5.28773880005
#### indic-en
#1_multilingual_zeroshot|phonetic|hi-en|010|1.29299092293
#1_multilingual_zeroshot|onehot_shared|hi-en|013|1.32769790292
### en-indic 
#1_multilingual_zeroshot|phonetic|en-hi|019|1.24506568909
#1_multilingual_zeroshot|onehot_shared|en-hi|017|1.24252215028


######################################################
##################### TRANSFER PIVOTING ################
######################################################
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname='2_pivoting_multilingual'
#representation='onehot_shared'
#
#spm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/1_multilingual_zeroshot/$representation/indic-en
#spm_prefix=13
#spm_prefix1=`echo $spm_prefix | sed 's,^0\+,,g'`
#
#ptm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015/1_multilingual_zeroshot/$representation/en-indic/
#ptm_prefix=17
#ptm_prefix1=`echo $ptm_prefix | sed 's,^0\+,,g'`
#
##for langpair `echo bn-hi bn-kn bn-ta hi-bn hi-kn hi-ta kn-bn kn-hi kn-ta ta-bn ta-hi ta-kn`
#for langpair in `echo bn-kn` 
#do 
#    echo 'Start: ' $dataset $expname $langpair $representation 
#    
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    echo $src_lang 
#    echo $tgt_lang 
#
#    o=$output_dir/$expname/$representation/indic-indic
#    rm -rf $o
#    mkdir -p $o/outputs
#    
#    output_1_fname="$o/outputs/001test.s1out.$src_lang-en-$tgt_lang.en"
#    
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair $src_lang-en \
#        --beam_size 5 \
#        --mapping_dir "$spm/mappings" \
#        --model_fname "$spm/temp_models/my_model-$spm_prefix1"  \
#        --representation "en:onehot,$src_lang:$representation" \
#        --enc_rnn_size 256 \
#        --dec_rnn_size 256 \
#        --max_seq_length 30 \
#        --in_fname    "$data_dir/$langpair/test/$langpair" \
#        --out_fname   $output_1_fname
#    
#    input_2nd_fname=$o/outputs/001test.s2in.$src_lang-en-$tgt_lang.en
#    sed  's/ ||| /|/g;s/ |/|/g' $output_1 | \
#        cut -d'|' -f2  > $input_2nd_fname
#    
#    output_2nd_fname="$o/outputs/001test.s2out.$src_lang-en-$tgt_lang.$tgt_lang"
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair en-$tgt_lang\
#        --beam_size 5 \
#        --mapping_dir "$ptm/mappings" \
#        --model_fname "$ptm/temp_models/my_model-$ptm_prefix1"  \
#        --representation "en:onehot,$tgt_lang:$representation" \
#        --enc_rnn_size 256 \
#        --dec_rnn_size 256 \
#        --max_seq_length 30 \
#        --in_fname   $input_2nd_fname \
#        --out_fname  $output_2nd_fname
#    
#    final_output_fname="$o/outputs/001test.nbest.$langpair.$tgt_lang"
#    python utilities.py transfer_pivot_translate $output_1_fname $output_2_fname $final_output_fname
#
#    echo 'End: ' $dataset $expname $langpair $representation 
#done 



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

