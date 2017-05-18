#!/bin/bash

export MLXLIT_BASE=/home/development/anoop/experiments/multilingual_unsup_xlit
export MLXLIT_HOME=$MLXLIT_BASE/src/multiling_unsup_xlit
export XLIT_HOME=$MLXLIT_BASE/src/conll16_unsup_xlit
export PYTHONPATH=$PYTHONPATH:$MLXLIT_HOME/src:$XLIT_HOME/src 
export INDIC_NLP_HOME=/home/development/anoop/installs/indic_nlp_library/

export CUDA_VISIBLE_DEVICES=0

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

#########################################################################
############  ZEROSHOT TRANSLITERATION WITH UNSEEN LANGUAGE #############
#########################################################################

#### common block
#src_lang=en
#tgt_lang=hi
#
#dataset='news_2015_official' 
##more_opts="--shared_mapping_class CharacterMapping"
#
#expname='2_multilingual_bn-kn'
#representation='onehot_shared' # for source language
#
#data_dir_moses=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/mosesformat/$dataset
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset/$expname/$representation/multi-conf

###################################################################################
####### baseline transliteration system: just use the output of proxy #############
###################################################################################
##for proxy_lang in `echo pl sk sl`
#for proxy_lang in `echo bn kn ta`
#do 
#    echo "***********************************"
#    echo "Using $proxy_lang as proxy language" 
#    echo "***********************************"
#    echo 
#
#    final_dir="$output_dir/proxy_for_${tgt_lang}/baseline/$proxy_lang"
#    mkdir -p $final_dir
#
#    ### find the best epoch for translation model
#    echo 'Find the best epoch for translation model'
#    x=`python utilities.py compute_accuracy \
#            $output_dir \
#            $src_lang $proxy_lang \
#            40`
#    best_trans_epoch=`echo $x | cut -f 1 -d '|' `
#    echo "Best translation model epoch: $best_trans_epoch" 
#
#    ### transliterate source to proxy 
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        --lang_pair $src_lang-$proxy_lang \
#        --beam_size 5 \
#        --mapping_dir "$output_dir/mappings" \
#        --model_fname "$output_dir/temp_models/my_model-$best_trans_epoch"  \
#        --representation "$src_lang:onehot,$proxy_lang:$representation" \
#        $more_opts \
#        --in_fname    "$data_dir/$src_lang-$tgt_lang/test/$src_lang-$tgt_lang" \
#        --out_fname   "$final_dir/test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang" 
#
#    ## postprocess to convert script to target language script 
#    ### just copy if the scripts are the same 
#    #cp "$final_dir/test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang"  \
#    #   "$final_dir/test.nbest.$src_lang-$tgt_lang.$tgt_lang" 
#   
#    ## or transliterate
#    python $INDIC_NLP_HOME/src/indicnlp/transliterate/unicode_transliterate.py \
#        transliterate \
#        "$final_dir/test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang"  \
#        "$final_dir/test.nbest.$src_lang-$tgt_lang.$tgt_lang" \
#        $proxy_lang $tgt_lang 
#
#    ## generate NEWS 2015 evaluation format output file 
#    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#            "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.id" \
#            "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.xml" \
#            "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#            "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#            "system" "conll2016" "$src_lang" "$tgt_lang"  
#    
#    # run evaluation 
#    python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#            -t "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.xml" \
#            -i "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#            -o "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#             > "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#
#
#done 

###################################################################################
####### baseline (enhanced)  system: post processing baseline ### #################
###################################################################################
#
#exp=baseline_enhanced
#
##for proxy_lang in `echo pl sk sl`
#for proxy_lang in `echo bn kn ta`
#do 
#    echo "***********************************"
#    echo "Using $proxy_lang as proxy language" 
#    echo "***********************************"
#    echo 
#
#    final_dir="$output_dir/proxy_for_${tgt_lang}/$exp/$proxy_lang"
#    mkdir -p $final_dir
#
#    python utilities.py remove_terminal_halant \
#        "$output_dir/proxy_for_${tgt_lang}/baseline/$proxy_lang/test.nbest.$src_lang-$tgt_lang.${tgt_lang}"  \
#        "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#        $tgt_lang
#
#    ## generate NEWS 2015 evaluation format output file 
#    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#            "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.id" \
#            "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.xml" \
#            "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#            "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#            "system" "conll2016" "$src_lang" "$tgt_lang"  
#    
#    # run evaluation 
#    python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#            -t "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.xml" \
#            -i "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#            -o "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#             > "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#
#done 

##################################################################################
######### transliterate with proxy and fuse with tgt language model #########
##################################################################################

### fixed lm weight experiment
##exp=lm_fusion
##best_lm_weight=0.2  ## a hyperparameter 
#
### fixed lm weight experiment
#exp=lm_fusion_oracle
#### weights are determined using parallel src-tgt validation set. Uncomment block in the loop to enable 
#
###for proxy_lang in `echo pl sk sl`
#for proxy_lang in `echo bn kn ta`
#do 
#    echo "***********************************"
#    echo "Using $proxy_lang as proxy language" 
#    echo "***********************************"
#    echo 
#
#    final_dir="$output_dir/proxy_for_${tgt_lang}/$exp/$proxy_lang"
#    lm_data_dir=$final_dir/lm_data_dir
#    lm_dir=$final_dir/lm_dir
#
#    mkdir -p $final_dir
#    mkdir -p $lm_data_dir 
#    mkdir $lm_dir
#    
#    ####### represent monolingual data of target language in the script of the proxy language #######
#
#    ## copy 
#    cp -r $data_dir_moses/$src_lang-$tgt_lang/*.$tgt_lang $lm_data_dir
#    ## transliteration not necessary for Indic since mapping using IndicPhoneticMapping is language neural
#
#    ############ train language model on the target language ########
#    #######
#    ## create LM for target language with same vocabulary as the translation model (proxy language) 
#    echo "Create LM for target language with same vocabulary as the translation model"
#
#    python $MLXLIT_HOME/src/unsup_xlit/LanguageModel.py \
#        train \
#        --lang       $tgt_lang \
#        --data_dir   $lm_data_dir \
#        --output_dir $lm_dir \
#        --representation onehot_shared \
#        --mapping_class  IndicPhoneticMapping \
#        --use_mapping $output_dir/mappings/mapping_${proxy_lang}.json \
#        --embedding_size 32 \
#        --rnn_size 32 \
#        --max_epochs 30 \
#    > $lm_dir/train.log 2>&1 
#
#    ######### find best epoch for LM ##############
#    echo 'Finding the best epoch for the LM' 
#    x=`python utilities.py early_stop_best \
#          loss \
#          30 \
#          $lm_dir/train.log`
#    best_lm_epoch=`echo $x | cut -f 1 -d '|' `
#    echo "Best LM epoch: $best_lm_epoch"
#
#    ####### find the best epoch for translation model  ##########
#    echo 'Find the best epoch for translation model'
#    x=`python utilities.py compute_accuracy \
#            $output_dir \
#            $src_lang $proxy_lang \
#            40`
#    best_trans_epoch=`echo $x | cut -f 1 -d '|' `
#    echo "Best translation model epoch: $best_trans_epoch" 
#
#    ####### #### find the best LM weights on validation set ##########
#    ### UNCOMMENT THIS BLOCK IF LM WEIGHTS HAVE TO BE DETERMINED #########
#    echo 'Find the best LM weight' 
#    mkdir -p $final_dir/validation_with_lm
#
#    for lm_weight in `echo 0.1 0.2 0.3 0.4 0.5`
#    do 
#        echo "Trying LM weight: $lm_weight" 
#
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecodingWithLm.py \
#            --lang_pair $src_lang-$proxy_lang \
#            --beam_size 5 \
#            --mapping_dir "$output_dir/mappings" \
#            --model_fname "$output_dir/temp_models/my_model-$best_trans_epoch"  \
#            --representation "$src_lang:onehot,$proxy_lang:$representation" \
#            $more_opts \
#            --fuse_lm $lm_dir/models/model-$best_lm_epoch \
#            --lm_weight $lm_weight \
#            --lm_embedding_size 32 \
#            --lm_rnn_size 32 \
#            --lm_max_seq_length 30 \
#            --in_fname  "$data_dir/$src_lang-$tgt_lang/parallel_valid/$src_lang-$tgt_lang.$src_lang" \
#            --out_fname   "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang" 
#
#        #### just copy if the scripts are the same 
#        #cp "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang" \
#        #"$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.$tgt_lang" 
#        
#        
#        ## or transliterate
#        python $INDIC_NLP_HOME/src/indicnlp/transliterate/unicode_transliterate.py \
#            transliterate \
#            "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang" \
#            "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.$tgt_lang" \
#            $proxy_lang $tgt_lang 
#
#        # generate NEWS 2015 evaluation format output file 
#        python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#                "$data_dir/$src_lang-$tgt_lang/parallel_valid/valid.$src_lang-$tgt_lang.id" \
#                "$data_dir/$src_lang-$tgt_lang/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
#                "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#                "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                "system" "conll2016" "$src_lang" "$tgt_lang" 
#    
#        # run evaluation 
#        python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#                -t "$data_dir/$src_lang-$tgt_lang/parallel_valid/valid.$src_lang-$tgt_lang.xml" \
#                -i "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#                -o "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#                 > "$final_dir/validation_with_lm/${lm_weight}_test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#    done 
#
#    x=`python utilities.py find_best_lm_weight \
#            $final_dir/validation_with_lm \
#            $src_lang $tgt_lang `
#    best_lm_weight=`echo $x | cut -f 1 -d '|' `
#    echo "Best LM weight: $best_lm_weight" 
#
#    ### END OF BLOCK #########
#
#    echo "Selected LM weight: $best_lm_weight"
#
#    ########## transliterate source to proxy #########
#    python $MLXLIT_HOME/src/unsup_xlit/ModelDecodingWithLm.py \
#        --lang_pair $src_lang-$proxy_lang \
#        --beam_size 5 \
#        --mapping_dir "$output_dir/mappings" \
#        --model_fname "$output_dir/temp_models/my_model-$best_trans_epoch"  \
#        --representation "$src_lang:onehot,$proxy_lang:$representation" \
#        $more_opts \
#        --fuse_lm $lm_dir/models/model-$best_lm_epoch \
#        --lm_weight $best_lm_weight \
#        --lm_embedding_size 32 \
#        --lm_rnn_size 32 \
#        --lm_max_seq_length 30 \
#        --in_fname    "$data_dir/$src_lang-$tgt_lang/test/$src_lang-$tgt_lang" \
#        --out_fname   "$final_dir/test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang" 
#
#    ########## postprocess to convert script to target language script ##########
#
#    #### just copy if the scripts are the same 
#    #cp "$final_dir/test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang"  \
#    #   "$final_dir/test.nbest.$src_lang-$tgt_lang.$tgt_lang" 
#    
#    ## or transliterate
#    python $INDIC_NLP_HOME/src/indicnlp/transliterate/unicode_transliterate.py \
#        transliterate \
#        "$final_dir/test.nbest.$src_lang-$tgt_lang.proxy.$tgt_lang"  \
#        "$final_dir/test.nbest.$src_lang-$tgt_lang.$tgt_lang" \
#        $proxy_lang $tgt_lang 
#
#    # generate NEWS 2015 evaluation format output file 
#    python $XLIT_HOME/src/cfilt/transliteration/news2015_utilities.py gen_news_output \
#            "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.id" \
#            "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.xml" \
#            "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}" \
#            "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#            "system" "conll2016" "$src_lang" "$tgt_lang"  
#    
#    # run evaluation 
#    python $XLIT_HOME/scripts/news_evaluation_script/news_evaluation.py \
#            -t "$data_dir/$src_lang-$tgt_lang/test/test.$src_lang-$tgt_lang.xml" \
#            -i "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.xml" \
#            -o "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.detaileval.csv" \
#             > "$final_dir/test.nbest.$src_lang-$tgt_lang.${tgt_lang}.eval"
#
#done 

############################
#######  ZEROSHOT: MANY-MANY
############################

### Decoding the validatio and test set for all epochs on the unseen language pairs 

#export NO_OUTEMBED=1
#
#dataset='news_2015_indic'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname="2_multilingual"
#representation="onehot_shared"
#
##for langpair in `echo en-hi en-bn en-kn en-ta`
##for langpair in `echo hi-en bn-en ta-en kn-en`
##for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
##for langpair in `echo pl-ar sk-ar sl-ar cs-ar `
##for langpair in `echo ar-sk ar-sl ar-pl`
#for langpair in `echo bn-ta ta-bn hi-kn kn-hi `
#do 
#
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#    o=$output_dir/$expname/$representation/multi-conf   #### Set this correctly for bilingual vs multilingual
#    
#    #### for every saved model, compute accuracy on the validation set 
#    ##### Note: Validation xml needs to be created 
#    
#    echo 'Start: ' $dataset $expname $langpair $representation 
#    
#    for prefix in `seq -f '%03g' 1 40`
#    do 
#        
#        prefix1=`echo $prefix | sed 's,^0\+,,g'`
#        echo $prefix  $prefix1     
#        
#        if [ $dataset = 'news_2015' -o $dataset = 'news_2015_official' ]
#        then 
#            ### for multilingual experiments  (en-indic)
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
#        ## outputs
#        #resdir='outputs'
#        #python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#        #    --lang_pair $langpair \
#        #    --beam_size 5 \
#        #    --mapping_dir "$o/mappings" \
#        #    --model_fname "$o/temp_models/my_model-$prefix1"  \
#        #    --representation $rep_str \
#        #    $more_opts \
#        #    --in_fname    "$data_dir/$langpair/test/$langpair" \
#        #    --out_fname   "$o/$resdir/${prefix}test.nbest.$langpair.$tgt_lang"
#
#        ## validation 
#        resdir='validation'
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#            --lang_pair $langpair \
#            --beam_size 5 \
#            --mapping_dir "$o/mappings" \
#            --model_fname "$o/temp_models/my_model-$prefix1"  \
#            --representation $rep_str \
#            $more_opts \
#            --in_fname    "$data_dir/$langpair/parallel_valid/$langpair.$src_lang" \
#            --out_fname   "$o/$resdir/${prefix}test.nbest.$langpair.$tgt_lang"
#    
#    done 
#    
#    echo 'End: ' $dataset $expname $langpair $representation 
#
#done 
#
#
