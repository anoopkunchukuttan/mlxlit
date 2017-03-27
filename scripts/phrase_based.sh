
###Building phrase-based systems

######   make LM  ###
#lm_dir=$MLXLIT_BASE/results/pbsmt/news_2015_official/lm/
#data_dir=$MLXLIT_BASE/data/sup/mosesformat/news_2015_official/
#
#for lang in `echo hi ta bn kn`
#do 
#    ngram-count -wbdiscount -interpolate \
#        -text $data_dir/en-$lang/train.$lang \
#        -lm $lm_dir/$lang.5g.lm \
#        -order 5 
#done 
    

###  train models

#parallel --joblog 'new2015official.job.log' --jobs 3 --gnu "nohup time /usr/local/bin/smt/moses_job_scripts/moses_run.sh run_params.en-{}.conf > en-{}.log 2>&1" <<  LANG_PAIRS
#bn
#hi
#kn
#ta
#LANG_PAIRS

## evaluate 

data_dir=$MLXLIT_BASE/data/sup/mosesformat/news_2015_official
o=$MLXLIT_BASE/results/pbsmt/news_2015_official

for langpair in `echo en-bn en-kn en-ta en-hi`
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
