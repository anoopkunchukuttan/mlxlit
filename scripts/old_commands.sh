
###########################################################################
########  Create gold standard xml files for the validation set also
###########################################################################

#dataset='slavic_latin-ar'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#
#for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
#do 
#
#    ####  for bilingual case 
#    #conf=$langpair
#    ###  for multilingual case 
#    conf='multi-conf'
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
    

##############################################################################################################################
## FINDING THE MODEL WITH THE BEST VALIDATION ACCURACY ########
##############################################################################################################################

#dataset='news_2015_official'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname="2_bilingual_bilstm"
#representation="onehot"
#
##for langpair in `echo en-hi en-bn en-kn en-ta`
##for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
#for langpair in `echo en-kn`
#do 
#
#    src_lang=`echo $langpair | cut -f 1 -d '-'`
#    tgt_lang=`echo $langpair | cut -f 2 -d '-'`
#    
#    o=$output_dir/$expname/$representation/$langpair   #### Set this correctly for bilingual vs multilingual
#    rep_str="en:onehot,$tgt_lang:$representation"     #### which dataset
#    #rep_str="$representation"     #### which dataset
#    
#    #### for every saved model, compute accuracy on the validation set 
#    ##### Note: Validation xml needs to be created 
#    
#    echo 'Start: ' $dataset $expname $langpair $representation 
#    
#    mkdir -p "$o/validation"
#    
#    for prefix in `seq -f '%03g' 1 40`
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
#            --enc_type bilstm \
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
#

##############################################################################################
###################### EVALUATE SPECIFIC ITERATION BY ENUMERATION ############################
##############################################################################################

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
#        ##### other multilingual experiments  
#        #o=$output_dir/$expname/$representation/multi-conf
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
#2_multilingual|onehot_shared|kn-hi|026|5.28773880005
#2_multilingual|onehot_shared|hi-kn|026|5.28773880005
#2_multilingual|onehot_shared|bn-ta|026|5.28773880005
#2_multilingual|onehot_shared|ta-bn|026|5.28773880005
#CONFIG

#2_multilingual_zeroshot|onehot_shared|kn-hi|013|4.71042811871
#2_multilingual_zeroshot|onehot_shared|hi-kn|013|4.71042811871
#2_multilingual_zeroshot|onehot_shared|bn-ta|013|4.71042811871
#2_multilingual_zeroshot|onehot_shared|ta-bn|013|4.71042811871

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



##################################
### DECODE TEST AND VALIDATION ###
##################################

#
#dataset='ar-slavic_latin'
#data_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/$dataset
#output_dir=/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/$dataset
#
#expname="2_multilingual_zeroshot"
#representation="onehot_shared"
#
##for langpair in `echo en-hi en-bn en-kn en-ta`
##for langpair in `echo hi-en bn-en ta-en kn-en`
##for langpair in `echo cs-ar pl-ar sk-ar sl-ar`
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn bn-ta ta-bn hi-kn kn-hi `
##for langpair in `echo bn-hi bn-kn hi-bn hi-ta kn-bn kn-ta ta-hi ta-kn`
##for langpair in `echo pl-ar sk-ar sl-ar cs-ar `
#for langpair in `echo ar-sk ar-sl ar-pl`
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
#        # outputs
#        resdir='outputs'
#        python $MLXLIT_HOME/src/unsup_xlit/ModelDecoding.py \
#            --lang_pair $langpair \
#            --beam_size 5 \
#            --mapping_dir "$o/mappings" \
#            --model_fname "$o/temp_models/my_model-$prefix1"  \
#            --representation $rep_str \
#            $more_opts \
#            --in_fname    "$data_dir/$langpair/test/$langpair" \
#            --out_fname   "$o/$resdir/${prefix}test.nbest.$langpair.$tgt_lang"
#
#        # validation 
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


