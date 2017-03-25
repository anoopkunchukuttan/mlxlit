#!/bin/bash 

## convert from mosesformat to one required by system

#### prepraring bilingual data 

inbase_parallel="/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/mosesformat/ar-slavic_latin"
outbase="/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/ar-slavic_latin"

for lang_pair in `echo ar-cs ar-sk ar-sl ar-pl`
do 
    src_lang=`echo "$lang_pair" | cut -f 1 -d '-' `
    tgt_lang=`echo "$lang_pair" | cut -f 2 -d '-' ` 

    outdir="$outbase/$lang_pair"

    indir_parallel="$inbase_parallel/$lang_pair"
    mkdir -p $outdir/{mono_train,parallel_train,parallel_valid,test}
    
    # copy train mono
    cp $indir_parallel/train.$src_lang $outdir/mono_train/$src_lang 
    cp $indir_parallel/train.$tgt_lang $outdir/mono_train/$tgt_lang 
    
    # copy test 
    cp $indir_parallel/test.$src_lang $outdir/test/$src_lang-$tgt_lang 
    cp $indir_parallel/test.$tgt_lang $outdir/test/$tgt_lang-$src_lang 
    
    # copy train parallel
    cp $indir_parallel/train.$src_lang $outdir/parallel_train/$lang_pair.$src_lang 
    cp $indir_parallel/train.$tgt_lang $outdir/parallel_train/$lang_pair.$tgt_lang 
    
    # copy valid parallel
    cp $indir_parallel/tun.$src_lang $outdir/parallel_valid/$lang_pair.$src_lang 
    cp $indir_parallel/tun.$tgt_lang $outdir/parallel_valid/$lang_pair.$tgt_lang 

    # copy xml evaluation files 

    cp $indir_parallel/test.xml $outdir/test/test.$src_lang-$tgt_lang.xml
    cp $indir_parallel/test.id $outdir/test/test.$src_lang-$tgt_lang.id

done 

####### prepraring multilingual data from bilingual data 

for lang_pair in `echo ar-cs ar-sk ar-sl ar-pl`
do 
    mkdir -p $outbase/multi-conf/{mono_train,parallel_train,parallel_valid,test}

    cp -r $outbase/$lang_pair/parallel_train/* $outbase/multi-conf/parallel_train/ 
    cp -r $outbase/$lang_pair/parallel_valid/* $outbase/multi-conf/parallel_valid/ 
    cp -r $outbase/$lang_pair/test/* $outbase/multi-conf/test/ 
    #cp -r $outbase/$lang_pair/mono_train/* $outbase/multi-conf/mono_train/ 
done

