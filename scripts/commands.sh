#!/bin/bash

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=$MLXLIT_BASE/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 

export CUDA_VISIBLE_DEVICES=2

###################################################################################################
################################ supervised transliteration - multilingual #########################
####################################################################################################

#dataset='ar-slavic_latin'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
##restore_epoch_number="33"
#
### Backward compatibility flags
##export NO_OUTEMBED=1
##export ALWAYS_LANG_TOKEN=1
#
#for expname in `echo 2_multilingual_zeroshot`
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
#            #lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "$src_lang-$x," ; done | sed 's/,$//g'`
#            #more_opts="${more_opts} --unseen_langs hi"
#            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf='multi-conf'
#            rep_str="$src_lang:onehot,$representations_param"
#
#        elif [ $dataset = 'news_2015_reversed' -o $dataset = 'news_2015_reversed_match' ]
#        then 
#            ## for INDIC-EN
#            more_opts=""
#
#            #### normal run
#            tgt_lang='en'
#            src_langs=(hi bn kn ta)
#            lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
#            representations_param=`for x in ${src_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ##### for zeroshot training
#            #tgt_lang='en'
#            #src_langs=(bn kn ta)
#            #all_langs=(hi bn kn ta)
#            #lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-$tgt_lang," ; done | sed 's/,$//g'`
#            #more_opts="${more_opts} --unseen_langs hi"
#            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf='multi-conf'
#            rep_str="$tgt_lang:onehot,$representations_param"
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
#            multiconf='multi-conf'
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
#            ##### normal run
#            #src_lang='ar'
#            #tgt_langs=(cs pl sl sk)
#            #lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "$src_lang-$x," ; done | sed 's/,$//g'`
#            #representations_param=`for x in ${tgt_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            #### for zeroshot training
#            src_lang='ar'
#            tgt_langs=(pl sl sk)
#            all_langs=(cs pl sl sk)
#            lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "$src_lang-$x," ; done | sed 's/,$//g'`
#            more_opts="${more_opts} --unseen_langs cs"
#            representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf='multi-conf'
#            rep_str="$src_lang:onehot,$representations_param"
#            more_opts="${more_opts} --shared_mapping_class CharacterMapping"
#
#        elif [ $dataset = 'slavic_latin-ar' ]
#        then 
#            # for SLAVIC_LATIN-AR
#            more_opts=""
#
#            #### normal run
#            tgt_lang='ar'
#            src_langs=(cs pl sl sk)
#            lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-$tgt_lang," ; done | sed 's/,$//g'`
#            representations_param=`for x in ${src_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ##### for zeroshot training
#            #tgt_lang='ar'
#            #src_langs=(pl sl sk)
#            #all_langs=(cs pl sl sk)
#            #lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-$tgt_lang," ; done | sed 's/,$//g'`
#            #more_opts="${more_opts} --unseen_langs cs"
#            #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#            ## common  block
#            multiconf='multi-conf'
#            rep_str="$tgt_lang:onehot,$representations_param"
#            more_opts="${more_opts} --shared_mapping_class CharacterMapping"
#        else
#            echo 'Invalid dataset' 
#            exit 1
#        fi 
#
#        o=$output_dir/$expname/$representation/$multiconf
#        
#        echo 'Start: ' $dataset $expname $multiconf $representation 
#    
#        #### Training and Testing 
#        rm -rf $o
#        mkdir -p $o
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            --lang_pairs "$lang_pairs" \
#            --data_dir  $data_dir/$multiconf \
#            --output_dir  $o \
#            --representation "$rep_str" \
#            --max_epochs 40 \
#             $more_opts >> $o/train.log 2>&1 
#    
#            #--start_from $restore_epoch_number \
#    
#        echo 'End: ' $dataset $expname $multiconf $representation 
#    
#    done 
#
#done 

##############################################################################################
######################### supervised transliteration - bilingual  ############################
###############################################################################################

#dataset='ar-slavic_latin'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
##restore_epoch_number="2"
#
### Backward compatibility flags
##export NO_OUTEMBED=1
##export ALWAYS_LANG_TOKEN=1
#
#for expname in `echo 2_multilingual_ar-pl-match`
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
#    for langpair in `echo ar-pl`
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
#            #### Training and Testing 
#            rm -rf $o
#            mkdir -p $o
#
#            python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#                --lang_pairs "$src_lang-$tgt_lang" \
#                --data_dir   "$data_dir/$langpair" \
#                --output_dir "$o" \
#                --representation "$rep_str" \
#                --max_epochs 40 \
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

#dataset='slavic_latin-ar'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#for prefix in `seq -f '%03g' 1 40`
#do 
#
#for expname in `echo 2_multilingual_prefix_src`
#do 
#
#    ######## Experiment loop starts here ########
#
#    #for langpair in `echo en-hi en-bn en-ta en-kn`
#    #for langpair in `echo hi-en bn-en ta-en kn-en`
#    #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
#    #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
#    for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
#    #for langpair in `echo ar-cs ar-pl ar-sk ar-sl`
#    do
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#        #for representation in `echo onehot phonetic`
#        for representation in `echo onehot_shared`
#        do 
#            #### output directory to select 
#            ### for bilingual experiments 
#            #o=$output_dir/$expname/$representation/$langpair
#
#            #### for other multilingual experiments  
#            o=$output_dir/$expname/$representation/multi-conf
#
#            echo 'Start: ' $dataset $expname $langpair $representation $prefix 
#    
#            #### Evaluation starts 
#
#            ####### ON TEST SET 
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
#            ####### ON VALIDATION SET 
#            resdir=validation
#            # generate NEWS 2015 evaluation format output file 
#            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                    "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.id" \
#                    "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
#                    "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#                    "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                    "system" "conll2016" "$src_lang" "$tgt_lang"  
#            
#            # run evaluation 
#            python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                    -t "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
#                    -i "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                    -o "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#                     > "$o/$resdir/${prefix}test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#
#            echo 'End: ' $dataset $expname $langpair $representation $prefix
#    
#        done 
#    done     
#done 
#done


####################################################################
#####################  FIND MININUM ITERATIONS AND SCORES ##########
####################################################################
#
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#for expname in `echo 100_test`
#do 
#
#    for representation in `echo onehot`
#    do 
#        ###### bilingual 
#        #for langpair in `echo en-hi en-bn en-ta en-kn`
#        #for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
#        #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
#        #for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
#        for langpair in `echo bn-hi`
#        do
#            src_lang=`echo $langpair | cut -f 1 -d '-'`
#            tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#
#            scores=`python utilities.py compute_accuracy \
#                $output_dir/$expname/$representation/$langpair \
#                $src_lang $tgt_lang \
#                40` 
#            echo "$dataset|$expname|$representation|$src_lang|$tgt_lang|$scores" 
#
#        done         
#
#       # ######### multilingual by averaging 
#       # echo "$dataset|$expname|$representation" 
#       # python utilities.py compute_accuracy_multilingual \
#       #         $output_dir/$expname/$representation/multi-conf \
#       #         40 
#
#    done         
#done     


########################################################################
################# Decoding #########################
########################################################################

######## for multilingual zeroshot training  (en-indic) with hindi as the missing language
#dataset='news_2015_reversed'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
### Backward compatibility flags
##export NO_OUTEMBED=1
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
#        ## bilingual
#        #o=$output_dir/$expname/$representation/$langpair
#
#        ## multilingual 
#        o=$output_dir/$expname/$representation/multi-conf
#
#        if [ $dataset = 'news_2015' -o $dataset = 'news_2015_official' ]
#        then 
#            ### for multilingual experiments  (en-indic)
#            #rep_str="en:onehot,$tgt_lang:$representation,bn:$representation,ta:$representation,kn:$representation"
#            rep_str="en:onehot,$tgt_lang:$representation"
#        elif [ $dataset = 'news_2015_reversed' ]
#        then 
#            ### for multilingual experiments  (indic-en)
#            rep_str="en:onehot,$src_lang:$representation"
#        elif [ $dataset = 'news_2015_indic' ]
#        then 
#            ##### for multilingual experiments  (indic-indic)
#            rep_str="$representation" 
#            if [ $representation = 'phonetic' ]
#            then 
#                more_opts="--separate_output_embedding"
#            fi 
#        elif [ $dataset = 'ar-slavic_latin' ]
#        then 
#            #  for multilingual experiments (AR-SLAVIC_LATIN)
#            rep_str="ar:onehot,$tgt_lang:$representation"
#            more_opts="--shared_mapping_class CharacterMapping"
#
#        elif [ $dataset = 'slavic_latin-ar' ]
#        then 
#            # for multilingual experiments (SLAVIC_LATIN-AR)
#            rep_str="ar:onehot,$src_lang:$representation"
#            more_opts="--shared_mapping_class CharacterMapping"
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
#            $more_opts \
#            --representation $rep_str \
#            --in_fname    "$data_dir/$langpair/test/$langpair" \
#            --out_fname   "$o/outputs/${prefix}test.nbest.$langpair.$tgt_lang"
#
#        echo 'End: ' $dataset $expname $langpair $representation 
#
#done  <<CONFIG
#CONFIG


#2_multilingual|phonetic|kn-hi|013|5.28773880005
#2_multilingual|phonetic|hi-kn|013|5.28773880005
#2_multilingual|phonetic|bn-ta|013|5.28773880005
#2_multilingual|phonetic|ta-bn|013|5.28773880005



#############################################################################################
############################################  VISUALIZATION #################################
#############################################################################################

########## for multilingual zeroshot training  (en-indic) with hindi as the missing language
#dataset='news_2015_official'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/mosesformat/$dataset
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
#        ## bilingual
#        o=$output_dir/$expname/$representation/$langpair
#
#        ## multilingual 
#        #o=$output_dir/$expname/$representation/multi-conf
#
#        if [ $dataset = 'news_2015' -o $dataset = 'news_2015_official' ]
#        then 
#            ### for multilingual experiments  (en-indic)
#            #rep_str="en:onehot,$tgt_lang:$representation,bn:$representation,ta:$representation,kn:$representation"
#            rep_str="en:onehot,$tgt_lang:$representation"
#        elif [ $dataset = 'news_2015_reversed' ]
#        then 
#            ### for multilingual experiments  (indic-en)
#            rep_str="en:onehot,$src_lang:$representation"
#        elif [ $dataset = 'news_2015_indic' ]
#        then 
#            ##### for multilingual experiments  (indic-indic)
#            rep_str="$tgt_lang:$representation,$src_lang:$representation"
#            if [ $representation = 'phonetic' ]
#            then 
#                more_opts="--separate_output_embedding"
#            fi 
#        elif [ $dataset = 'ar-slavic_latin' ]
#        then 
#            #  for multilingual experiments (AR-SLAVIC_LATIN)
#            rep_str="ar:onehot,$tgt_lang:$representation"
#            more_opts="--shared_mapping_class CharacterMapping"
#
#        elif [ $dataset = 'slavic_latin-ar' ]
#        then 
#            # for multilingual experiments (SLAVIC_LATIN-AR)
#            rep_str="ar:onehot,$src_lang:$representation"
#            more_opts="--shared_mapping_class CharacterMapping"
#        else
#            echo 'Invalid dataset'
#            exit 1
#        fi 
#
#        echo 'Start: ' $dataset $expname $langpair $representation 
#
#        python $MLXLIT_HOME/src/unsup_xlit/encoder_analysis.py \
#                --lang $src_lang \
#                --representation "$rep_str" \
#                --model_fname "$o/temp_models/my_model-$prefix1"  \
#                --mapping_dir "$o/mappings" \
#                --window_size 1 \
#                --in_fname $data_dir/$langpair/test.$src_lang \
#                --out_img_fname $o/outputs/${prefix}_analysis_$langpair/encoder_rep.png \
#                --out_html_fname $o/outputs/${prefix}_analysis_$langpair/encoder_rep.html
#
#        echo 'End: ' $dataset $expname $langpair $representation 
#
#done  <<CONFIG
#CONFIG

#2_bilingual|onehot|en-hi|026|0.641
#2_bilingual|onehot|en-bn|021|0.417
#2_bilingual|onehot|en-ta|038|0.578
#2_bilingual|onehot|en-kn|025|0.52

#2_multilingual|onehot_shared|en-hi|023|0.607
#2_multilingual|onehot_shared|en-bn|022|0.461
#2_multilingual|onehot_shared|en-ta|026|0.553
#2_multilingual|onehot_shared|en-kn|040|0.539

#2_bilingual|onehot|hi-en|022|0.382591
#2_bilingual|onehot|bn-en|032|0.48934
#2_bilingual|onehot|ta-en|023|0.232232
#2_bilingual|onehot|kn-en|036|0.337675


#2_multilingual|onehot_shared|hi-en|037|0.511134
#2_multilingual|onehot_shared|bn-en|022|0.540102
#2_multilingual|onehot_shared|ta-en|031|0.259259
#2_multilingual|onehot_shared|kn-en|022|0.476954



######################################################
##################### TRANSFER PIVOTING ################
######################################################


######### Bilingual or Multilingual: see the inline notes for configuration ####
##
#
##  DECODING

#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname='2_indic_pivoting_multilingual'
#representation='onehot_shared'
#    
#for langtriple in `echo bn-hi-ta ta-hi-bn hi-ta-kn kn-ta-hi bn-kn-ta ta-kn-bn hi-bn-kn kn-bn-hi `
#do 
#    echo 'Start: ' $dataset $expname $langtriple $representation 
#    
#    src_lang=`echo $langtriple | cut -f 1 -d '-'`
#    pvt_lang=`echo $langtriple | cut -f 2 -d '-'`
#    tgt_lang=`echo $langtriple | cut -f 3 -d '-'`
#
#    langpair="$src_lang-$tgt_lang"
#
#    o=$output_dir/$expname/$representation/$langtriple
#    echo 'Output directory: ' $o
#    rm -rf $o
#    mkdir -p $o/outputs
#
#    ## NOTE: for multilingual - change the directory name 
#    ##bilingual
#    #spm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset/2_bilingual/$representation/$src_lang-$pvt_lang
#    #spm_prefix=`python utilities.py read_best_epoch results_with_accuracy.csv $dataset 2_bilingual $representation $src_lang $pvt_lang`
#    ##multilingual
#    spm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset/2_multilingual/$representation/multi-conf
#    spm_prefix=`python utilities.py read_best_epoch results_with_accuracy.csv $dataset 2_multilingual $representation $src_lang $pvt_lang`
#    spm_prefix1=`echo $spm_prefix | sed 's,^0\+,,g'`
#    echo 'Epoch used for source-pivot model: ' $spm_prefix
#
#    ## NOTE:for multilingual - change the directory name 
#    ## bilingual
#    #ptm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset/2_bilingual/$representation/$pvt_lang-$tgt_lang
#    #ptm_prefix=`python utilities.py read_best_epoch results_with_accuracy.csv $dataset 2_bilingual $representation $pvt_lang $tgt_lang`
#    ## multilingual
#    ptm=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset/2_multilingual/$representation/multi-conf
#    ptm_prefix=`python utilities.py read_best_epoch results_with_accuracy.csv $dataset 2_multilingual $representation $pvt_lang $tgt_lang`
#    ptm_prefix1=`echo $ptm_prefix | sed 's,^0\+,,g'`
#    echo 'Epoch used for pivot-target model: ' $ptm_prefix
#
#    output_1_fname="$o/outputs/001test.s1out.$src_lang-$pvt_lang-$tgt_lang.$pvt_lang"
#    echo 'Output src-pvt transliteration: ' $output_1_fname
#    
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair $src_lang-$pvt_lang \
#        --beam_size 5 \
#        --mapping_dir "$spm/mappings" \
#        --model_fname "$spm/temp_models/my_model-$spm_prefix1"  \
#        --representation "$pvt_lang:$representation,$src_lang:$representation" \
#        --batch_size  100 \
#        --in_fname    "$data_dir/$langpair/test/$langpair" \
#        --out_fname   $output_1_fname
#    
#    input_2_fname=$o/outputs/001test.s2in.$src_lang-$pvt_lang-$tgt_lang.$pvt_lang
#    echo 'Input pvt-tgt transliteration: ' $input_2_fname
#    sed  's/ ||| /|/g;s/ |/|/g' $output_1_fname | \
#        cut -d'|' -f2  > $input_2_fname
#    
#    output_2_fname="$o/outputs/001test.s2out.$src_lang-$pvt_lang-$tgt_lang.$tgt_lang"
#    echo 'Output pvt-tgt transliteration: ' $output_2_fname
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair $pvt_lang-$tgt_lang\
#        --beam_size 5 \
#        --mapping_dir "$ptm/mappings" \
#        --model_fname "$ptm/temp_models/my_model-$ptm_prefix1"  \
#        --representation "$tgt_lang:$representation,$pvt_lang:$representation" \
#        --batch_size  100 \
#        --in_fname   $input_2_fname \
#        --out_fname  $output_2_fname
#    
#    final_output_fname="$o/outputs/001test.nbest.$langpair.$tgt_lang"
#    echo 'Final output: ' $final_output_fname
#    python utilities.py transfer_pivot_translate $output_1_fname $output_2_fname $final_output_fname
#
#    echo 'End: ' $dataset $expname $langtriple $representation 
#    echo
#done 
#
#
##### EVALUATION 
#
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#for prefix in `seq -f '%03g' 1 1`
#do 
#
#for expname in `echo 2_indic_pivoting_multilingual`
#do 
#
#    ######## Experiment loop starts here ########
#
#    for langtriple in `echo bn-hi-ta ta-hi-bn hi-ta-kn kn-ta-hi bn-kn-ta ta-kn-bn hi-bn-kn kn-bn-hi `
#    do
#        src_lang=`echo $langtriple | cut -f 1 -d '-'`
#        pvt_lang=`echo $langtriple | cut -f 2 -d '-'`
#        tgt_lang=`echo $langtriple | cut -f 3 -d '-'`
#
#        langpair="$src_lang-$tgt_lang"
#    
#        for representation in `echo phonetic`
#        do 
#            #### output directory to select 
#            o=$output_dir/$expname/$representation/$langtriple
#
#            echo 'Start: ' $dataset $expname $langpair $representation $prefix 
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
#            echo 'End: ' $dataset $expname $langpair $representation $prefix
#    
#        done 
#    done     
#done 
#done
#

##########################################################
################ LANGUAGE MODELLING #####################
##########################################################

###### discovering optimal hyperparameters 
#lang=en
#representation=onehot 
#
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/lm_data/$lang
#outdir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/lm/$lang

### training lms

##for esize in `echo 16 32 64 128 256`
#for esize in `echo 8`
#do 
#    #for rsize in `echo 32 64 128 256 512`
#    for rsize in `echo 8`
#    do 
#        echo $lang $representation $esize $rsize 
#
#        o=$outdir/e_$esize-r_$rsize
#        rm -rf $o
#        mkdir -p $o
#
#        python $MLXLIT_HOME/src/unsup_xlit/LanguageModel.py \
#            train \
#            --lang $lang \
#            --representation $representation \
#            --mapping_class CharacterMapping \
#            --max_epochs 50 \
#            --data_dir  $data_dir \
#            --output_dir $o \
#            --embedding_size $esize \
#            --rnn_size       $rsize \
#        > $o/train.log 2>&1 & 
#
#    done 
#done 

#### finding best architectures 
#
#for er in `echo 8-8 16-16 32-32 32-64`
#do 
#    esize=`echo $er | cut -f 1 -d '-'`
#    rsize=`echo $er | cut -f 2 -d '-'`
#
#    echo -n $lang $representation $esize $rsize " "
#    
#    python utilities.py early_stop_best \
#           loss \
#           50 \
#           $outdir/e_$esize-r_$rsize/train.log
#done 
#
#for esize in `echo 64 128 256`
#do 
#    for rsize in `echo 64 128 256 512`
#    do 
#        echo -n $lang $representation $esize $rsize " "
#        python utilities.py early_stop_best \
#              loss \
#              50 \
#              $outdir/e_$esize-r_$rsize/train.log
#    done
#done 

#mapping_fname="/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/2_multilingual_match/onehot_shared/multi-conf/mappings/mapping_en.json"
#
#outdir=out_map
#
#### train 
#rm -rf $outdir
#python $MLXLIT_HOME/src/unsup_xlit/LanguageModel.py \
#    train \
#    --lang en \
#    --data_dir  indir \
#    --output_dir $outdir \
#    --representation onehot \
#    --mapping_class CharacterMapping \
#    --embededding_size 256 \
#    --rnn_size 512 \
#    --max_epochs 50 \
#> $outdir/train.log 2>&1 
    
#    --use_mapping $mapping_fname \

###### test 
#python $MLXLIT_HOME/src/unsup_xlit/LanguageModel.py \
#    "test" \
#    --lang en \
#    --in_fname  indir/test.txt \
#    --model_fname $outdir/models/model-2 \
#    --representation onehot \
#    --mapping_fname $mapping_fname \
#    --mapping_class CharacterMapping \
#> $outdir.log 2>&1 


##### debug lm

#python $MLXLIT_HOME/src/unsup_xlit/LanguageModel.py \
#    "print_vars" \
#    --lang en \
#    --model_fname $outdir/models/model-2 \
#    --representation onehot \
#    --mapping_fname $mapping_fname \
#    --mapping_class CharacterMapping

######## decoding with LM

#dataset='news_2015_reversed'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#o=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/2_multilingual_match/onehot_shared/multi-conf
#
#python $MLXLIT_HOME/src/unsup_xlit/ModelDecodingWithLm.py \
#    --lang_pair hi-en \
#    --beam_size 2 \
#    --mapping_dir "$o/mappings" \
#    --model_fname "$o/temp_models/my_model-20"  \
#    --representation 'hi:onehot_shared,en:onehot' \
#    --fuse_lm $outdir/models/model-2 \
#    --lm_weight 0.5 \
#    --in_fname    "$data_dir/multi-conf/test/hi-en" \
#    --out_fname   out.en

    #--in_fname    "indir/test.txt" \


########################################################################
###########  RERANKING TRANSLITERATION OUTPUT WITH LM ##################
#########################################################################

### Note: these experiments are being done for indic-en and slavic-ar,
## where the target language has access to a larger target data for 
## multilingual model. Hence, we use larger LMs for the bilingual models

dataset='news_2015_reversed' 
export NO_OUTEMBED=1
expname='2_bilingual'
representation='onehot' # for source language
tgt_lang='en'

lm_data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/lm_data/
data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset

#for src_lang in `echo bn kn ta hi`
for src_lang in `echo hi`
do 
    langpair=$src_lang-$tgt_lang
    o=$output_dir/$expname/$representation/$langpair
    logf="$o/xlit_with_lm.log "

    echo "Experiment for $langpair starts" >> $logf

    #######
    ## create LM for target language with same vocabulary as the translation model 
    echo "Create LM for target language with same vocabulary as the translation model"  >> $logf

    lm_dir=$o/lm_dir

    mkdir $lm_dir
    python $MLXLIT_HOME/src/unsup_xlit/LanguageModel.py \
        train \
        --lang $tgt_lang \
        --data_dir  $lm_data_dir/$tgt_lang \
        --output_dir $lm_dir \
        --representation onehot \
        --mapping_class CharacterMapping \
        --use_mapping $o/mappings/mapping_${tgt_lang}.json \
        --embedding_size 32 \
        --rnn_size 32 \
        --max_epochs 30 \
    > $lm_dir/train.log 2>&1 

    #######
    ## find best epoch for LM
    echo 'Finding the best epoch for the LM'  >> $logf
    x=`python utilities.py early_stop_best \
          loss \
          30 \
          $lm_dir/train.log`
    best_lm_epoch=`echo $x | cut -f 1 -d '|' `
    echo "Best LM epoch: $best_lm_epoch" >> $logf

    #######
    ## find the best epoch for translation model
    echo 'Find the best epoch for translation model' >> $logf
    x=`python utilities.py compute_accuracy \
            $o \
            $src_lang $tgt_lang \
            40`
    best_trans_epoch=`echo $x | cut -f 1 -d '|' `
    echo "Best translation model epoch: $best_trans_epoch" >> $logf

    #######
    #### find the best LM weights on validation set 
    echo 'Find the best LM weight' >> $logf
    mkdir -p $o/validation_with_lm

    for lm_weight in `echo 0.1 0.2 0.3 0.4 0.5`
    do 
        echo "Trying LM weight: $lm_weight" >> $logf

        python $MLXLIT_HOME/src/unsup_xlit/ModelDecodingWithLm.py \
            --lang_pair $langpair \
            --beam_size 5 \
            --mapping_dir "$o/mappings" \
            --model_fname "$o/temp_models/my_model-$best_trans_epoch"  \
            --representation "$src_lang:$representation,$tgt_lang:onehot" \
            --fuse_lm $lm_dir/models/model-$best_lm_epoch \
            --lm_weight $lm_weight \
            --lm_embedding_size 32 \
            --lm_rnn_size 32 \
            --lm_max_seq_length 30 \
            --in_fname  "$data_dir/$langpair/parallel_valid/$langpair.$src_lang" \
            --out_fname   "$o/validation_with_lm/${lm_weight}_test.nbest.$langpair.$tgt_lang" >> $logf

        # generate NEWS 2015 evaluation format output file 
        python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
                "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.id" \
                "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
                "$o/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
                "system" "conll2016" "$src_lang" "$tgt_lang"  >> $logf
    
        # run evaluation 
        python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
                -t "$data_dir/$langpair/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
                -i "$o/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
                -o "$o/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
                 > "$o/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
    done 

    x=`python utilities.py find_best_lm_weight \
            $o/validation_with_lm \
            $src_lang $tgt_lang `
    best_lm_weight=`echo $x | cut -f 1 -d '|' `
    echo "Best LM weight: $best_lm_weight" >> $logf

    #######
    ## transliterate the output with LM for the optimal LM weights learnt in the previous step
    echo "Transliterate with LM using optimal LM weight" >> $logf
    mkdir -p "$o/outputs_with_lm"
    python $MLXLIT_HOME/src/unsup_xlit/ModelDecodingWithLm.py \
        --lang_pair $langpair \
        --beam_size 5 \
        --mapping_dir "$o/mappings" \
        --model_fname "$o/temp_models/my_model-$best_trans_epoch"  \
        --representation "$src_lang:$representation,$tgt_lang:onehot" \
        --fuse_lm $lm_dir/models/model-$best_lm_epoch \
        --lm_weight $best_lm_weight \
        --lm_embedding_size 32 \
        --lm_rnn_size 32 \
        --lm_max_seq_length 30 \
        --in_fname    "$data_dir/$langpair/test/$langpair" \
        --out_fname   "$o/outputs_with_lm/test.nbest.$langpair.$tgt_lang" >> $logf

    # generate NEWS 2015 evaluation format output file 
    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
            "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.id" \
            "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
            "$o/outputs_with_lm/test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
            "$o/outputs_with_lm/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
            "system" "conll2016" "$src_lang" "$tgt_lang"  >> $logf  
    
    # run evaluation 
    python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
            -t "$data_dir/$langpair/test/test.$src_lang-$tgt_lang.xml" \
            -i "$o/outputs_with_lm/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
            -o "$o/outputs_with_lm/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
             > "$o/outputs_with_lm/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"

done 
