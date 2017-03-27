#!/bin/bash

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=$MLXLIT_BASE/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 

export CUDA_VISIBLE_DEVICES=0

###################################################################################################
################################ supervised transliteration - multilingual #########################
####################################################################################################

#dataset='news_2015_official'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#restore_epoch_number="20"
#
#for expname in `echo 2_multilingual`
#do 
#
#    ######## Experiment loop starts here ########
#
#    #for representation in `echo onehot phonetic`
#    for representation in `echo onehot_shared`
#    do 
#
#        if [ $dataset = 'news_2015' -o $dataset = 'news_2015_official' ]
#        then 
#            # for EN-INDIC
#
#            #### normal run
#            src_lang='en'
#            tgt_langs=(hi bn kn ta)
#            lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
#            representations_param=`for x in ${tgt_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            #### for zeroshot training
#            #src_lang='en'
#            #tgt_langs=(bn kn ta)
#            #all_langs=(hi bn kn ta)
#            #lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
#            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf=en-indic
#            rep_str="en:onehot,$representations_param"
#
#        elif [ $dataset = 'news_2015_reversed' ]
#        then 
#            ## for INDIC-EN
#
#            #### normal run
#            tgt_lang='en'
#            src_langs=(hi bn kn ta)
#            lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
#            representations_param=`for x in ${src_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            #### for zeroshot training
#            #tgt_lang='en'
#            #src_langs=(bn kn ta)
#            #all_langs=(hi bn kn ta)
#            #lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
#            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf=indic-en
#            rep_str="en:onehot,$representations_param"
#
#        elif [ $dataset = 'news_2015_indic' ]
#        then
#            ## for INDIC-INDIC 
#
#            ## all pairs 
#            #lang_pairs="bn-hi,bn-kn,bn-ta,hi-bn,hi-kn,hi-ta,kn-bn,kn-hi,kn-ta,ta-bn,ta-hi,ta-kn"
#        
#            #some language pairs: then separate run for zeroshot is not required 
#            lang_pairs="bn-hi,bn-kn,hi-bn,hi-ta,kn-bn,kn-ta,ta-hi,ta-kn"
#
#            ## common block 
#            multiconf=indic-indic
#            rep_str="$representation"
#            if [ $representation = 'phonetic' ]
#            then 
#                more_opts="--separate_output_embedding"
#            fi 
#
#        elif [ $dataset = 'ar-slavic_latin' ]
#        then 
#            # for AR-SLAVIC_LATIN
#
#            #### normal run
#            src_lang='ar'
#            tgt_langs=(cs pl sl sk)
#            lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "$src_lang-$x," ; done | sed 's/,$//g'`
#            representations_param=`for x in ${tgt_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf='multi-conf'
#            rep_str="$src_lang:onehot,$representations_param"
#            more_opts="--shared_mapping_class CharacterMapping"
#
#        elif [ $dataset = 'slavic_latin-ar' ]
#        then 
#            # for SLAVIC_LATIN-AR
#
#            #### normal run
#            tgt_lang='ar'
#            src_langs=(cs pl sl sk)
#            lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-$tgt_lang," ; done | sed 's/,$//g'`
#            representations_param=`for x in ${src_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf='multi-conf'
#            rep_str="$tgt_lang:onehot,$representations_param"
#            more_opts="--shared_mapping_class CharacterMapping"
#        else
#            echo 'Invalid dataset' 
#            exit 1
#        fi 
#
#        o=$output_dir/$expname/$representation/$multiconf
#        
#        echo 'Start: ' $dataset $expname $multiconf $representation 
#    
#        ### Training and Testing 
#        #rm -rf $o
#        #mkdir -p $o
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            --lang_pairs "$lang_pairs" \
#            --data_dir  $data_dir/$multiconf \
#            --output_dir  $o \
#            --representation "$rep_str" \
#            --max_epochs 30 \
#            --start_from $restore_epoch_number \
#             $more_opts >> $o/train.log 2>&1 
#    
#    
#        echo 'End: ' $dataset $expname $multiconf $representation 
#    
#    done 
#
#done 

##############################################################################################
######################### supervised transliteration - bilingual  ############################
##############################################################################################
#
#dataset='news_2015_official'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#restore_epoch_number="30"
#
#for expname in `echo 2_bilingual`
#do 
#
#    ######## Experiment loop starts here ########
#
#    #for langpair in `echo en-hi en-bn en-kn en-ta`
#    #for langpair in `echo hi-en bn-en kn-en ta-en`
#    #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
#    #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
#    #for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
#    #for langpair in `echo ar-cs ar-pl ar-sk ar-sl`
#    #for langpair in `echo en-hi`
#    #for langpair in `echo en-bn`
#    for langpair in `echo en-hi`
#    #for langpair in `echo en-ta`
#    do
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#        #for representation in `echo onehot phonetic`
#        for representation in `echo onehot`
#        do 
#            o=$output_dir/$expname/$representation/$langpair
#            
#            echo 'Start: ' $dataset $expname $langpair $representation 
#
#            if [ $dataset = 'news_2015' -o $dataset = 'news_2015_official' ]
#            then 
#                rep_str="en:onehot,$tgt_lang:$representation"
#            elif [ $dataset = 'news_2014_reversed' ]
#            then 
#                rep_str="en:onehot,$src_lang:$representation"
#            elif [ $dataset = 'news_2015_indic' ]
#            then 
#                rep_str="$representation"
#                if [ $representation = 'phonetic' ]
#                then 
#                    more_opts="--separate_output_embedding"
#                fi 
#            elif [ $dataset = 'ar-slavic_latin' ]
#            then 
#                rep_str="ar:onehot,$tgt_lang:$representation"
#                more_opts="--shared_mapping_class CharacterMapping"
#            elif [ $dataset = 'slavic_latin-ar' ]
#            then 
#                rep_str="ar:onehot,$src_lang:$representation"
#                more_opts="--shared_mapping_class CharacterMapping"
#            else
#                echo 'Invalid dataset' 
#                exit 1
#            fi 
#    
#            ### Training and Testing 
#            #rm -rf $o
#            #mkdir -p $o
#
#            python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#                --lang_pairs "$src_lang-$tgt_lang" \
#                --data_dir   "$data_dir/$langpair" \
#                --output_dir "$o" \
#                --representation "$rep_str" \
#                --max_epochs 40 \
#                --start_from $restore_epoch_number \
#                $more_opts \
#                >> $o/train.log 2>&1 
#    
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

dataset='news_2015_official'
data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset

for prefix in `seq -f '%03g' 20 30`
do 

for expname in `echo 2_multilingual`
do 

    ######## Experiment loop starts here ########

    for langpair in `echo en-hi en-bn en-ta en-kn`
    #for langpair in `echo hi-en bn-en ta-en kn-en`
    #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
    do
        src_lang=`echo $langpair | cut -f 1 -d '-'`
        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
    
        #for representation in `echo onehot phonetic`
        for representation in `echo onehot_shared`
        do 
            #### output directory to select 
            ### for bilingual experiments 
            #o=$output_dir/$expname/$representation/$langpair

            ### for multilingual experiments  (en-indic)
            o=$output_dir/$expname/$representation/en-indic

            #### for multilingual experiments  (indic-en)
            #o=$output_dir/$expname/$representation/indic-en
            
            #### for multilingual experiments  (indic-indic)
            #o=$output_dir/$expname/$representation/indic-indic

            echo 'Start: ' $dataset $expname $langpair $representation $prefix 
    
            #### Evaluation starts 
            
            resdir=outputs
    
            # generate NEWS 2015 evaluation format output file 
            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
                    "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.id" \
                    "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
                    "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
                    "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
                    "system" "conll2016" "$src_lang" "$tgt_lang"  
            
            # run evaluation 
            python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
                    -t "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
                    -i "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
                    -o "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
                     > "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
    
            echo 'End: ' $dataset $expname $langpair $representation $prefix
    
        done 
    done     
done 
done

#dataset='news_2015_official'
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
#        ##### for multilingual experiments  (indic-en)
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
#2_bilingual|onehot|en-ta|001|0.45933803916
#2_bilingual|onehot|en-ta|002|0.45933803916
#2_bilingual|onehot|en-ta|003|0.45933803916
#2_bilingual|onehot|en-ta|004|0.45933803916
#2_bilingual|onehot|en-ta|005|0.45933803916
#2_bilingual|onehot|en-ta|006|0.45933803916
#2_bilingual|onehot|en-ta|007|0.45933803916
#2_bilingual|onehot|en-ta|008|0.45933803916
#2_bilingual|onehot|en-ta|009|0.45933803916
#2_bilingual|onehot|en-ta|010|0.45933803916
#2_bilingual|onehot|en-ta|011|0.45933803916
#2_bilingual|onehot|en-ta|012|0.45933803916
#2_bilingual|onehot|en-ta|013|0.45933803916
#2_bilingual|onehot|en-ta|014|0.45933803916
#2_bilingual|onehot|en-ta|015|0.45933803916
#2_bilingual|onehot|en-ta|016|0.45933803916
#2_bilingual|onehot|en-ta|017|0.45933803916
#2_bilingual|onehot|en-ta|018|0.45933803916
#2_bilingual|onehot|en-ta|019|0.45933803916
#CONFIG

#2_multilingual_prefix_src|onehot_shared|hi-en|006|0.458529800177
#2_multilingual_prefix_src|onehot_shared|kn-en|006|0.458529800177
#2_multilingual_prefix_src|onehot_shared|bn-en|006|0.458529800177
#2_multilingual_prefix_src|onehot_shared|ta-en|006|0.458529800177

#2_multilingual|onehot_shared|bn-hi|010|4.71042811871
#2_multilingual|onehot_shared|bn-kn|010|4.71042811871
#2_multilingual|onehot_shared|hi-bn|010|4.71042811871
#2_multilingual|onehot_shared|hi-ta|010|4.71042811871
#2_multilingual|onehot_shared|kn-bn|010|4.71042811871
#2_multilingual|onehot_shared|kn-ta|010|4.71042811871
#2_multilingual|onehot_shared|ta-hi|010|4.71042811871
#2_multilingual|onehot_shared|ta-kn|010|4.71042811871

########################################################################
################# Decoding #########################
########################################################################

###### for multilingual zeroshot training  (en-indic) with hindi as the missing language
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
#            #rep_str="en:onehot,$tgt_lang:$representation,bn:$representation,ta:$representation,kn:$representation"
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
#            --out_fname test.hi
#            #--out_fname   "$o/outputs/${prefix}test.nbest.$langpair.$tgt_lang"
#
#        echo 'End: ' $dataset $expname $langpair $representation 
#
#done  <<CONFIG
#2_multilingual|phonetic|bn-hi|009|5.28773880005
#CONFIG
#2_multilingual|phonetic|hi-kn|009|5.28773880005
#2_multilingual|phonetic|bn-ta|009|5.28773880005
#2_multilingual|phonetic|ta-bn|009|5.28773880005

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


##############################################################################################################################
#################################### various experiments to improve accuracy:   ##############################################
###################### Experiments done on en-indic bilingual, with news_2015_official dataset ###############################
##############################################################################################################################

##### (a) increasing beam size 
#
#dataset='news_2015_official'
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
#        o=$output_dir/$expname/$representation/$src_lang-$tgt_lang
#        rep_str="en:onehot,$tgt_lang:$representation"
#
#        echo 'Start: ' $dataset $expname $langpair $representation 
#
#        newo=$output_dir/${expname}-a/$representation/$src_lang-$tgt_lang/
#        mkdir -p $newo/outputs/
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#            --lang_pair $langpair \
#            --beam_size 10 \
#            --mapping_dir "$o/mappings" \
#            --model_fname "$o/temp_models/my_model-$prefix1"  \
#            --representation $rep_str \
#            --in_fname    "$data_dir/$langpair/test/$langpair" \
#            --out_fname   "$newo/outputs/${prefix}test.nbest.$langpair.$tgt_lang"
#
#        # generate NEWS 2015 evaluation format output file 
#        python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.id" \
#                "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
#                "$newo/outputs/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#                "$newo/outputs/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                "system" "conll2016" "$src_lang" "$tgt_lang"  
#        
#        # run evaluation 
#        python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                -t "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
#                -i "$newo/outputs/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                -o "$newo/outputs/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#                 > "$newo/outputs/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#        echo 'End: ' $dataset $expname $langpair $representation 
#
#done  <<CONFIG
#2_bilingual|onehot|en-kn|009|5.27229315042
#2_bilingual|onehot|en-bn|007|5.27229315042
#CONFIG


##### (b) Check if selecting best model using validation accuracy is is better than using validation loss 

#dataset='news_2015_official'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname="2_multilingual"
#representation="onehot_shared"
#
#for langpair in `echo en-hi en-bn en-kn en-ta`
#do 
#
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#    o=$output_dir/$expname/$representation/en-indic #### Set this correctly for bilingual vs multilingual
#    rep_str="en:onehot,$tgt_lang:$representation"
#    
#    #### for every saved model, compute accuracy on the validation set 
#    ##### Note: Validation xml needs to be created 
#    
#    echo 'Start: ' $dataset $expname $langpair $representation 
#    
#    mkdir -p "$o/validation"
#    
#    for prefix in `seq -f '%03g' 20 30`
#    do 
#        
#        prefix1=`echo $prefix | sed 's,^0\+,,g'`
#        echo $prefix  $prefix1     
#        
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#            --lang_pair $langpair \
#            --beam_size 5 \
#            --mapping_dir "$o/mappings" \
#            --model_fname "$o/temp_models/my_model-$prefix1"  \
#            --representation $rep_str \
#            --in_fname    "$data_dir/$langpair/parallel_valid/$langpair.$src_lang" \
#            --out_fname   "$o/validation/${prefix}test.nbest.$langpair.$tgt_lang"
#        
#        # generate NEWS 2015 evaluation format output file 
#        python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.id" \
#                "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
#                "$o/validation/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#                "$o/validation/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                "system" "conll2016" "$src_lang" "$tgt_lang"  
#        
#        # run evaluation 
#        python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                -t "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
#                -i "$o/validation/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                -o "$o/validation/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#                 > "$o/validation/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#    
#    done 
#    
#    echo 'End: ' $dataset $expname $langpair $representation 
#
#done 

###################################################################################################
##################### find the best model for an experiment  #########################
####################################################################################################

#find 2_bilingual_bilstm -name 'train.log' | \
    #while read line ; 
#do 
#    echo -n $line " "
#    awk -F ' ' 'BEGIN{min_v=1000.0;min_l=""} /Validation loss/{ if(min_v>$7){min_v=$7; min_l=$0;}} END{print min_l}' $line 
#done 


#### put the lowest loss iteration number in a file in the experiment directiory 
#find 4_* -name 'train.log' | \ 
#while read line ; 
#do 
#    min_iter=`awk -F ' ' 'BEGIN{min_v=1000.0;min_l=""} /Validation loss/{ if(min_v>$7){min_v=$7; min_l=$0;}} END{print min_l}' $line  | \
#            cut -f 2 -d ':'  | sed 's,Validation loss,,g' | tr -d ' '`
#    echo $min_iter >  `dirname $line`/min_loss_iter.modout
#done 

###################################################################################################
##################### error analysis  #########################
####################################################################################################

##### perform alignment of output 
#
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
##for langpair in `echo en-hi en-bn en-kn en-ta`
##for langpair in `echo hi-en bn-en kn-en ta-en`
#for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
##for langpair in `echo en-bn en-kn en-ta`
#do 
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    ## multilingual
#    #od="$output_dir/2_multilingual/onehot_shared/indic-en"
#    #od="$output_dir/2_multilingual/onehot_shared/en-indic"
#    od="$output_dir/2_multilingual/onehot_shared/indic-indic"
#    ## bilingual
#    #od="$output_dir/2_bilingual/onehot/$src_lang-$tgt_lang"
#
#    best_model=`cat $od/min_loss_iter.modout`
#
#    echo ${od} ${best_model}
#
#    ### generate 1 best output 
#    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py convert_to_1best_format \
#        ${od}/outputs/${best_model}test.nbest.$src_lang-$tgt_lang.$tgt_lang  \
#        ${od}/outputs/${best_model}test.1best.$src_lang-$tgt_lang.$tgt_lang  
#
#    ### generate analysis information 
#    analysisdir="${od}/outputs/${best_model}analysisdir_$langpair"
#    mkdir -p $analysisdir 
#    python $MLXLIT_BASE/src/conll16_unsup_xlit/src/cfilt/transliteration/analysis/align.py  \
#        $data_dir/$langpair/test/$tgt_lang-$src_lang \
#        ${od}/outputs/${best_model}test.1best.$src_lang-$tgt_lang.$tgt_lang \
#        $tgt_lang \
#        $analysisdir            
#
#done 

###########  TRAIN A MULTILINGUAL MODEL MATCHING THE SIZE OF HINDI CORPUS FOR EN-HI
#
#
## for EN-INDIC
#dataset='news_2015'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname='2_multilingual'
#representation='onehot_shared'
#
##### normal run
#src_lang='en'
#tgt_langs=(hi bn kn ta)
#lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
#representations_param=`for x in ${tgt_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
### common  block
#multiconf=en-indic_match-hi
#rep_str="en:onehot,$representations_param"
#
#o=$output_dir/$expname/$representation/$multiconf
#
#echo 'Start: ' $dataset $expname $multiconf $representation 
#
#### Training and Testing 
#rm -rf $o
#mkdir -p $o
#
#python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#    --lang_pairs "$lang_pairs" \
#    --data_dir  $data_dir/$multiconf \
#    --output_dir  $o \
#    --representation "$rep_str" \
#    --max_epochs 20 \
#     $more_opts >> $o/train.log 2>&1 
#
#echo 'End: ' $dataset $expname $multiconf $representation 
    
######################################################
##################### TRANSFER PIVOTING ################
######################################################

####### Multilingual ####

#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname='2_pivoting_multilingual'
#representation='onehot_shared'
#
#spm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/2_multilingual/$representation/indic-en
#spm_prefix=011
#spm_prefix1=`echo $spm_prefix | sed 's,^0\+,,g'`
#
#ptm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015/2_multilingual/$representation/en-indic
#ptm_prefix=006
#ptm_prefix1=`echo $ptm_prefix | sed 's,^0\+,,g'`
#
#o=$output_dir/$expname/$representation/indic-indic
#rm -rf $o
#mkdir -p $o/outputs
#    
#for langpair in `echo bn-hi bn-kn bn-ta hi-bn hi-kn hi-ta kn-bn kn-hi kn-ta ta-bn ta-hi ta-kn`
#do 
#    echo 'Start: ' $dataset $expname $langpair $representation 
#    
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    output_1_fname="$o/outputs/001test.s1out.$src_lang-en-$tgt_lang.en"
#    
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair $src_lang-en \
#        --beam_size 5 \
#        --mapping_dir "$spm/mappings" \
#        --model_fname "$spm/temp_models/my_model-$spm_prefix1"  \
#        --representation "en:onehot,$src_lang:$representation" \
#        --batch_size  100 \
#        --in_fname    "$data_dir/$langpair/test/$langpair" \
#        --out_fname   $output_1_fname
#    
#    input_2_fname=$o/outputs/001test.s2in.$src_lang-en-$tgt_lang.en
#    sed  's/ ||| /|/g;s/ |/|/g' $output_1_fname | \
#        cut -d'|' -f2  > $input_2_fname
#    
#    output_2_fname="$o/outputs/001test.s2out.$src_lang-en-$tgt_lang.$tgt_lang"
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair en-$tgt_lang\
#        --beam_size 5 \
#        --mapping_dir "$ptm/mappings" \
#        --model_fname "$ptm/temp_models/my_model-$ptm_prefix1"  \
#        --representation "en:onehot,$tgt_lang:$representation" \
#        --batch_size  100 \
#        --in_fname   $input_2_fname \
#        --out_fname  $output_2_fname
#    
#    final_output_fname="$o/outputs/001test.nbest.$langpair.$tgt_lang"
#    python utilities.py transfer_pivot_translate $output_1_fname $output_2_fname $final_output_fname
#
#    echo 'End: ' $dataset $expname $langpair $representation 
#done 

######## Bilingual ####
#
#function get_spm_epoch(){
#
#  if [ $1 == 'kn' ]
#  then 
#      echo '006'
#  else 
#      echo '009'
#  fi 
#
#}
#
#function get_ptm_epoch(){
#
#  if   [ $1 == 'hi' ]
#  then 
#      echo '008'
#  elif [ $1 == 'bn' ]
#  then 
#      echo '008'
#  elif [ $1 == 'kn' ]
#  then 
#      echo '010'
#  elif [ $1 == 'ta' ]
#  then 
#      echo '009'
#  fi 
#}
#
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname='2_pivoting_bilingual'
#representation='onehot'
#    
#for langpair in `echo bn-hi bn-kn bn-ta hi-bn hi-kn hi-ta kn-bn kn-hi kn-ta ta-bn ta-hi ta-kn`
#do 
#    echo 'Start: ' $dataset $expname $langpair $representation 
#    
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#    o=$output_dir/$expname/$representation/$langpair
#    rm -rf $o
#    mkdir -p $o/outputs
#
#    spm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/2_bilingual/$representation/$src_lang-en
#    spm_prefix=`get_spm_epoch $src_lang`
#    spm_prefix1=`echo $spm_prefix | sed 's,^0\+,,g'`
#    
#    ptm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015/2_bilingual/$representation/en-$tgt_lang
#    ptm_prefix=`get_ptm_epoch $tgt_lang`
#    ptm_prefix1=`echo $ptm_prefix | sed 's,^0\+,,g'`
#
#    output_1_fname="$o/outputs/001test.s1out.$src_lang-en-$tgt_lang.en"
#    
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair $src_lang-en \
#        --beam_size 5 \
#        --mapping_dir "$spm/mappings" \
#        --model_fname "$spm/temp_models/my_model-$spm_prefix1"  \
#        --representation "en:onehot,$src_lang:$representation" \
#        --batch_size  100 \
#        --in_fname    "$data_dir/$langpair/test/$langpair" \
#        --out_fname   $output_1_fname
#    
#    input_2_fname=$o/outputs/001test.s2in.$src_lang-en-$tgt_lang.en
#    sed  's/ ||| /|/g;s/ |/|/g' $output_1_fname | \
#        cut -d'|' -f2  > $input_2_fname
#    
#    output_2_fname="$o/outputs/001test.s2out.$src_lang-en-$tgt_lang.$tgt_lang"
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair en-$tgt_lang\
#        --beam_size 5 \
#        --mapping_dir "$ptm/mappings" \
#        --model_fname "$ptm/temp_models/my_model-$ptm_prefix1"  \
#        --representation "en:onehot,$tgt_lang:$representation" \
#        --batch_size  100 \
#        --in_fname   $input_2_fname \
#        --out_fname  $output_2_fname
#    
#    final_output_fname="$o/outputs/001test.nbest.$langpair.$tgt_lang"
#    python utilities.py transfer_pivot_translate $output_1_fname $output_2_fname $final_output_fname
#
#    echo 'End: ' $dataset $expname $langpair $representation 
#done 


###############
########  Create gold standard xml files for the validation set also
##############
#
#dataset='news_2015_official'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#
#for langpair in `echo en-ta en-bn en-kn en-hi`
#do 
#
#    ####  for bilingual case 
#    conf=$langpair
#    ###  for multilingual case 
#    #conf='en-indic'
#
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#    size=`wc -l "$data_dir/$conf/parallel_valid/$langpair.$src_lang" | cut -f 1 -d " "`
#
#    #### create the validation set xml gold standard 
#    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py generate_news_2015_gold_standard  \
#        "$data_dir/$conf/parallel_valid/$langpair.$src_lang"  \
#        "$data_dir/$conf/parallel_valid/$langpair.$tgt_lang"  \
#        "$data_dir/$conf/parallel_valid/valid.$langpair.xml"  \
#         'validation set' $size $src_lang $tgt_lang
#
#    seq 1 $size  | sed 's,$,_1_0,g' > \
#                 "$data_dir/$conf/parallel_valid/valid.$langpair.id"
#    
#done 

