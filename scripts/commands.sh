#!/bin/bash 

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=$MLXLIT_BASE/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 

export CUDA_VISIBLE_DEVICES=1

#############################################################################################
########################## supervised transliteration - multilingual #########################
##############################################################################################

#dataset='news_2015'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
##restore_epoch_number="17"
#
#for expname in `echo 1_multilingual_decshared_dec1024`
#do 
#
#    exptype=`echo $expname | cut -f 1 -d '_'`
#
#    ######### choose specific model  ############
#
#    if [ $exptype = '1' ]  # sup_nomono
#    then 
#        train_bidirectional=''
#        use_monolingual=''
#    elif [ $exptype = '2' ] # bisup_nomono 
#    then 
#        train_bidirectional='--train_bidirectional'
#        use_monolingual=''
#    elif [ $exptype = '3' ] # sup_mono 
#    then 
#        train_bidirectional=''
#        use_monolingual='--use_monolingual'
#    elif [ $exptype = '4' ] # bisup_mono
#    then 
#        train_bidirectional='--train_bidirectional'
#        use_monolingual='--use_monolingual'
#    fi 
#   
#    ######## Experiment loop starts here ########
#
#    #for representation in `echo onehot phonetic`
#    for representation in `echo onehot_shared`
#    do 
#
#        # for EN-INDIC
#        src_lang='en'
#        tgt_langs=(hi bn kn ta)
#        multiconf=en-indic
#        lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
#        representations_param=`for x in ${tgt_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#        ## for EN-INDIC  (for zeroshot training)
#        #src_lang='en'
#        #tgt_langs=(bn kn ta)
#        #all_langs=(hi bn kn ta)
#        #multiconf=en-indic
#        #lang_pairs=`for x in ${tgt_langs[*]}; do echo -n  "en-$x," ; done | sed 's/,$//g'`
#        #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#        ### for INDIC-EN
#        #tgt_lang='en'
#        #src_langs=(hi bn kn ta)
#        #multiconf=indic-en
#        #lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
#        #representations_param=`for x in ${src_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#        ### for INDIC-EN (for zeroshot training)
#        #tgt_lang='en'
#        #src_langs=(bn kn ta)
#        #all_langs=(hi bn kn ta)
#        #multiconf=indic-en
#        #lang_pairs=`for x in ${src_langs[*]}; do echo -n  "$x-en," ; done | sed 's/,$//g'`
#        #representations_param=`for x in ${all_langs[*]}; do echo -n  "$x:$representation," ; done | sed 's/,$//g'`
#
#        o=$output_dir/$expname/$representation/$multiconf
#        
#        echo 'Start: ' $expname $multiconf $representation 
#    
#        ### Training and Testing 
#        rm -rf $o
#        mkdir -p $o
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            --train_mode sup \
#            $train_bidirectional \
#            $use_monolingual \
#            --lang_pairs "$lang_pairs" \
#            $more_opts \
#            --data_dir  $data_dir/$multiconf \
#            --output_dir  $o \
#            --dec_rnn_size 1024 \
#            --representation "en:onehot,$representations_param"  \
#            --max_epochs 30 >> $o/train.log 2>&1 
#    
#            #--start_from $restore_epoch_number \
#    
#        echo 'End: ' $expname $langpair $representation 
#    
#    done 
#
#done 

#############################################################################################
########################## supervised transliteration - multilingual INDIC-INDIC #########################
##############################################################################################

#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
##restore_epoch_number="17"
#
#for expname in `echo 1_multilingual_sep_ioembed`
#do 
#
#    exptype=`echo $expname | cut -f 1 -d '_'`
#
#    ######### choose specific model  ############
#
#    if [ $exptype = '1' ]  # sup_nomono
#    then 
#        train_bidirectional=''
#        use_monolingual=''
#    elif [ $exptype = '2' ] # bisup_nomono 
#    then 
#        train_bidirectional='--train_bidirectional'
#        use_monolingual=''
#    elif [ $exptype = '3' ] # sup_mono 
#    then 
#        train_bidirectional=''
#        use_monolingual='--use_monolingual'
#    elif [ $exptype = '4' ] # bisup_mono
#    then 
#        train_bidirectional='--train_bidirectional'
#        use_monolingual='--use_monolingual'
#    fi 
#   
#    ######## Experiment loop starts here ########
#
#    #for representation in `echo onehot phonetic`
#    for representation in `echo phonetic`
#    do 
#
#        multiconf=indic-indic
#
#        ## all pairs 
#        #lang_pairs="bn-hi,bn-kn,bn-ta,hi-bn,hi-kn,hi-ta,kn-bn,kn-hi,kn-ta,ta-bn,ta-hi,ta-kn"
#
#        ##for zero-shot removed some pairs 
#        lang_pairs="bn-hi,bn-kn,hi-bn,hi-ta,kn-bn,kn-ta,ta-hi,ta-kn"
#        ## lpairs to be used for evaluation: kn-hi,hi-kn,bn-ta,ta-bn,
#        o=$output_dir/$expname/$representation/$multiconf
#        
#        echo 'Start: ' $expname $multiconf $representation 
#    
#        ### Training and Testing 
#        rm -rf $o
#        mkdir -p $o
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            --train_mode sup \
#            $train_bidirectional \
#            $use_monolingual \
#            --lang_pairs "$lang_pairs" \
#            --data_dir  $data_dir/$multiconf \
#            --output_dir  $o \
#            --representation "$representation"  \
#            --batch_size 32 \
#            --max_epochs 30 >> $o/train.log 2>&1 
#    
#            #--start_from $restore_epoch_number \
#    
#        echo 'End: ' $expname $langpair $representation 
#    
#    done 
#
#done 

#############################################################################################
######################## supervised transliteration ########################################
#############################################################################################

dataset='news_2015_reversed'
data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset

#restore_epoch_number="17"

#for expname in `echo 1_sup_nomono 2_bisup_nomono 3_sup_mono 4_bisup_mono`
#for expname in `echo 1_sup_nomono 2_bisup_nomono 3_sup_mono 4_bisup_mono 3_2_use_src 3_3_use_tgt 4_2_all_loss 4_3_ll_rep_loss`
for expname in `echo 1_test_2`
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
        use_monolingual='--use_monolingual'
    elif [ $exptype = '4' ] # bisup_mono
    then 
        train_bidirectional='--train_bidirectional'
        use_monolingual='--use_monolingual'
    fi 
   
    ######## Experiment loop starts here ########

    #for langpair in `echo hi-kn bn-hi ta-kn`
    #for langpair in `echo hi-en bn-en kn-en ta-en`
    #for langpair `echo bn-hi bn-kn bn-ta hi-bn hi-kn hi-ta kn-bn kn-hi kn-ta ta-bn ta-hi ta-kn`
    for langpair in `echo hi-en`
    do
        src_lang=`echo $langpair | cut -f 1 -d '-'`
        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
    
        #for representation in `echo onehot phonetic`
        for representation in `echo onehot`
        do 
            o=$output_dir/$expname/$representation/$langpair
            
            echo 'Start: ' $expname $langpair $representation 
    
            ### Training and Testing 
            rm -rf $o
            mkdir -p $o

            python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
                --train_mode sup \
                $train_bidirectional \
                $use_monolingual \
                --lang_pairs "$src_lang-$tgt_lang" \
                --data_dir  $data_dir/$langpair \
                --output_dir  $o \
                --batch_size 32 \
                --representation $representation \
                --max_epochs 30 >> $o/train.log 2>&1 
    
                #--representation "en:onehot,$src_lang:$representation"  \
                #--start_from $restore_epoch_number \

            echo 'End: ' $expname $langpair $representation 
    
        done 
    done     

done 

#############################################################################################
##################### EVALUATE SPECIFIC ITERATION #################
#############################################################################################

#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#for prefix in `echo  014`
#do 
#
#for expname in `echo 1_multilingual_shared_decoder`
#do 
#
#    ######## Experiment loop starts here ########
#
#    #for langpair in `echo en-hi en-bn en-ta en-kn`
#    #for langpair in `echo hi-en bn-en ta-en kn-en`
#    for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
#    do
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#        #for representation in `echo onehot phonetic`
#        for representation in `echo phonetic onehot_shared`
#        do 
#            #### output directory to select 
#            ### for bilingual experiments 
#            #o=$output_dir/$expname/$representation/$langpair
#
#            ### for multilingual experiments  (en-indic)
#            #o=$output_dir/$expname/$representation/en-indic
#
#            #### for multilingual experiments  (indic-en)
#            #o=$output_dir/$expname/$representation/indic-en
#            
#            #### for multilingual experiments  (indic-indic)
#            o=$output_dir/$expname/$representation/indic-indic
#
#            echo 'Start: ' $expname $langpair $representation 
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
#            echo 'End: ' $expname $langpair $representation 
#    
#        done 
#    done     
#done 
#done

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
#        echo $e_r_l_p_v
#        echo 
#
#        #### output directory to select 
#        #### for bilingual experiments 
#        #o=$output_dir/$expname/$representation/$langpair
#
#        #### for multilingual experiments  (en-indic)
#        #o=$output_dir/$expname/$representation/en-indic
#
#        #### for multilingual experiments  (indic-en)
#        #o=$output_dir/$expname/$representation/indic-en
#        
#        ##### for multilingual experiments  (indic-indic)
#        o=$output_dir/$expname/$representation/indic-indic
#
#        echo 'Start: ' $expname $langpair $representation 
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
#        echo 'End: ' $expname $langpair $representation 
#done  <<CONFIG
#1_multilingual_shared_decoder|phonetic|kn-hi|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|hi-kn|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|bn-ta|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|ta-bn|014|5.28773880005
#CONFIG

#1_multilingual_shared_decoder|onehot_shared|kn-hi|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|hi-kn|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|bn-ta|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|ta-bn|014|5.27229315042

########################################################################
################# Unsupervised transliteration #########################
#########################################################################

#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/conll16
### refdir contains same ref as data_dir, but in XML format required for evaluation tools. These are from the CoNLL 2016 directories
#ref_dir=~/experiments/unsupervised_transliterator/data/nonparallel/pb
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/unsup/conll16
#
#for expname in `echo 1_again`
#do 
#
#    for langpair in `echo hi-kn bn-hi ta-kn`
#    do
#        src_lang=`echo $langpair | cut -f 1 -d '-'`
#        tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#        for representation in `echo onehot phonetic`
#        do 
#            o=$output_dir/$expname/$representation/$langpair
#            
#            echo 'Start: ' $expname $langpair $representation 
#    
#            ### Training and Testing 
#            #rm -rf $o
#            #mkdir -p $o
#            #python $MLXLIT_HOME/src/unsup_xlit/ModelTraining.py \
#            #    --train_mode unsup \
#            #    --langs "$src_lang,$tgt_lang" \
#            #    --data_dir  $data_dir/$langpair \
#            #    --output_dir  $o \
#            #    --representation $representation > $o/train.log 2>&1 
#    
#            #### Evaluation starts 
#            
#            resdir=final_output
#            prefix=`ls $o/$resdir/ | sed 's,[^0-9],,g' | sort -r -n | head -1`
#    
#            ## convert to required format 
#            python utilities.py convert_output_format  \
#                $o/${resdir}/${prefix}${src_lang}-${tgt_lang}_ \
#                $o/${resdir}/${prefix}test.${tgt_lang} 
#    
#            # convert to n-best format 
#            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py  convert_to_nbest_format  \
#                $o/$resdir/${prefix}test.${tgt_lang}  $o/$resdir/${prefix}test.nbest.${tgt_lang}
#            
#            # generate NEWS 2015 evaluation format output file 
#            python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                    "$ref_dir/$src_lang-$tgt_lang/test.id" \
#                    "$ref_dir/$src_lang-$tgt_lang/test.xml" \
#                    "$o/$resdir/${prefix}test.nbest.${tgt_lang}" \
#                    "$o/$resdir/${prefix}test.nbest.${tgt_lang}.xml" \
#                    "system" "conll2016" "$src_lang" "$tgt_lang"  
#            
#            # run evaluation 
#            python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                    -t "$ref_dir/$src_lang-$tgt_lang/test.xml" \
#                    -i "$o/$resdir/${prefix}test.nbest.${tgt_lang}.xml" \
#                    -o "$o/$resdir/${prefix}test.nbest.${tgt_lang}.detaileval.csv" \
#                     > "$o/$resdir/${prefix}test.nbest.${tgt_lang}.eval"
#    
#            echo 'End: ' $expname $langpair $representation 
#    
#        done 
#    done     
#
#
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

########################################################################
################# Semi supervised transliteration #########################
########################################################################
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

########################################################################
################# Decoding #########################
########################################################################

#### for multilingual zeroshot training  (en-indic) with hindi as the missing language
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
#        else
#            true 
#        fi 
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#            --lang_pair $langpair \
#            --beam_size 5 \
#            --mapping_dir "$o/mappings" \
#            --model_fname "$o/temp_models/my_model-$prefix1"  \
#            --representation $rep_str \
#            --in_fname    "$data_dir/$langpair/test/$langpair" \
#            --out_fname   "$o/outputs/${prefix}test.nbest.$langpair.$tgt_lang"
#
#done  <<CONFIG
#1_multilingual_shared_decoder|onehot_shared|kn-hi|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|hi-kn|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|bn-ta|014|5.27229315042
#1_multilingual_shared_decoder|onehot_shared|ta-bn|014|5.27229315042
#1_multilingual_shared_decoder|phonetic|kn-hi|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|hi-kn|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|bn-ta|014|5.28773880005
#1_multilingual_shared_decoder|phonetic|ta-bn|014|5.28773880005
#CONFIG

## indic-indic
#1_multilingual_shared_decoder|onehot_shared||014|5.27229315042
#1_multilingual_shared_decoder|phonetic||014|5.28773880005
#1_multilingual_sep_ioembed|phonetic|bn-hi|017|5.18011403084
# indic-en
#1_multilingual_zeroshot|phonetic|hi-en|010|1.29299092293
#1_multilingual_zeroshot|onehot_shared|hi-en|013|1.32769790292
## en-indic 
#1_multilingual_zeroshot|phonetic|en-hi|019|1.24506568909
#1_multilingual_zeroshot|onehot_shared|en-hi|017|1.24252215028

