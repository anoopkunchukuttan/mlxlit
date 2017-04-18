import numpy as np
import pandas as pd 
import sys, os, codecs 

import matplotlib
import matplotlib.pyplot as plt

from cfilt.transliteration.analysis import align
from cfilt.transliteration import news2015_utilities as nwutil

from indicnlp.script import indic_scripts as isc 
from indicnlp.transliterate import unicode_transliterate  as indtrans

def get_column_name(x,tlang): 
    if isc.is_supported_language(tlang): 
        #return x if tlang=='hi' else indtrans.UnicodeIndicTransliterator.transliterate(x,tlang,'hi')
        if isc.in_coordinated_range(x,tlang): 
            return indtrans.ItransTransliterator.to_itrans(x,tlang) + '({:2x})'.format(isc.get_offset(x,tlang))
        else: 
            return str(hex(ord(x)))
    elif tlang=='ar': 
        pass 
        return x
    else: 
        return x

def plot_confusion_matrix(confusion_df,tlang,image_fname):
    """
    Plots a heat map of the confusion matrix of character alignments. 
    
    confusion_mat_fname: The input is a confusion matrix generated by the align script (csv format)
    tgt: target language of the transliteration
    
    For Indic scripts, the heat map shows characters in Devanagiri irrespective of target language for readability
        - Needs 'Lohit Devanagari' font to be installed 
    """
    
    #matplotlib.rc('font', family='Lohit Kannada') 
    matplotlib.rcParams.update({'font.size': 8})


    schar=list(confusion_df.index)
    tchar=list(confusion_df.columns)
    i=0
    for c in schar: 
        if c in tchar: 
            confusion_df.ix[c,c]=0.0
    
    data=confusion_df.as_matrix()
    
    columns=list(confusion_df.columns)
    col_names=[ get_column_name(x,tlang) for x in columns]

    rows=list(confusion_df.index)
    row_names=[ get_column_name(x,tlang) for x in rows]
    
    plt.figure(figsize=(20,10))

    plt.pcolor(data,cmap=plt.cm.gray_r,edgecolors='k')
    #plt.pcolor(data,cmap=plt.cm.hot_r,edgecolors='k')
    
    plt.colorbar()
    plt.xticks(np.arange(0,len(col_names))+0.5,col_names,rotation='vertical')
    plt.yticks(np.arange(0,len(row_names))+0.5,row_names)
    plt.xlabel('system')
    plt.ylabel('reference')
    
    plt.savefig(image_fname)
    plt.close()

def transliteration_analysis(exp_dirname,epoch,ref_fname,slang,tlang):
    """
    exp_dirname: base directory of the experiment 
    epoch: model to use - indicated by the epoch number 
    ref_fname: reference file 
    slang: source langauge 
    tlang: target language 
    """
   
    ## generate file names 
    out_fname='{exp_dirname}/outputs/{epoch:03d}test.nbest.{slang}-{tlang}.{tlang}'.format(
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)
    out1b_fname='{exp_dirname}/outputs/{epoch:03d}test.1best.{slang}-{tlang}.{tlang}'.format(
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)

    ## save the output 
    nwutil.convert_to_1best_format(out_fname,out1b_fname)
    out_dirname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}'.format(
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)
    align.save_analysis_artifacts(ref_fname, out1b_fname, tlang, out_dirname)

    ## plot the confusion matrix  
    confmat_fname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}/confusion_mat.pickle'.format( 
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang) 
    confmat_img_fname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}/confusion_mat.png'.format( 
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang) 

    conf_mat=pd.read_pickle(confmat_fname)
    plot_confusion_matrix(conf_mat,tlang,confmat_img_fname)

def transliteration_comparison(exp_dirname1,epoch1,
                exp_dirname2,epoch2,
                ref_fname,
                slang,tlang,
                out_fname):

    confmat_1_fname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}/confusion_mat.pickle'.format( 
        exp_dirname=exp_dirname1,epoch=epoch1,slang=slang,tlang=tlang) 
    confmat_1_df=pd.read_pickle(confmat_1_fname)

    confmat_2_fname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}/confusion_mat.pickle'.format( 
        exp_dirname=exp_dirname2,epoch=epoch2,slang=slang,tlang=tlang) 
    confmat_2_df=pd.read_pickle(confmat_2_fname)

    diff_conf_mat=  confmat_2_df.subtract(confmat_1_df,fill_value=0.0)#.divide(confmat_2_df,fill_value=0.0)
    diff_conf_mat=diff_conf_mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    plot_confusion_matrix(diff_conf_mat,tlang,out_fname)

########  Commands #############

# krishna 
krishna_datasets=[
                     'ar-slavic_latin',
                     'news_2015_reversed',
        ]

# balaram
balaram_datasets=[
                     'slavic_latin-ar',
                     'news_2015_official',
                     'news_2015_indic',
        ]

def get_edir(rec): 
    """
     get the name of the basedir for the experiment 
    """

    if (rec['exp'].find('bilingual')>=0) or (rec['exp'].find('moses')>=0) : 
        return '{}-{}'.format(rec['src'],rec['tgt'])

    elif rec['exp'].find('multilingual')>=0: 
        return 'multi-conf'
        #if rec['dataset'] in ['ar-slavic_latin','slavic_latin-ar']: 
        #    return 'multi-conf'
        #elif rec['dataset'] == 'news_2015_official' : 
        #    return 'en-indic'
        #elif rec['dataset'] == 'news_2015_reversed' : 
        #    return 'indic-en'
        #elif rec['dataset'] == 'news_2015_indic' : 
        #    return 'indic-indic'
        #else: 
        #    print 'Unknown experiment' 

    else: 
        print 'Invalid configuration'


def run_comparison_bi_multi(basedir,exp_conf_fname,out_dirname): 

    ## read the list of experiments to be analyzed 
    print 'Read list of experiments' 
    conf_df=pd.read_csv(exp_conf_fname,header=0,sep=',')

    print 'Comparing bilingual vs multilingual experiments ' 
    datasets=[]


    print 'Getting to work ' 
    for dataset in datasets: 

        print '** Experiments for dataset: *** ' + dataset 

        ## multilingual
        multi_df=conf_df.query('dataset=="{dataset}" & exp=="2_multilingual" & representation=="onehot_shared"'.format(dataset=dataset))
        ## bilingual
        bi_df=conf_df.query('dataset=="{dataset}" & exp=="2_bilingual" & representation=="onehot"'.format(dataset=dataset))

        for multi_rec in  [ x[1] for x in multi_df.iterrows() ]: 

            slang,tlang=(multi_rec['src'],multi_rec['tgt'])

            bi_rec=bi_df.query('src=="{}" & tgt=="{}"'.format(slang,tlang))
            bi_rec=bi_rec.iterrows().next()[1]

            multi_dirname = '{basedir}/results/sup/{dataset}/2_multilingual/onehot_shared/{edir}'.format(
                    basedir=basedir,dataset=dataset,edir=get_edir(multi_rec))
            bi_dirname = '{basedir}/results/sup/{dataset}/2_bilingual/onehot/{edir}'.format(
                    basedir=basedir,dataset=dataset,edir=get_edir(bi_rec))

            ref_fname = '{basedir}/data/sup/mosesformat/{dataset}/{slang}-{tlang}/test.{tlang}'.format(
                basedir=basedir,dataset=dataset,slang=slang,tlang=tlang)

            out_fname = '{}/{}-{}-{}.png'.format(out_dirname,dataset,slang,tlang)

            print 'Starting: {} {} {}'.format(dataset,slang,tlang)

            transliteration_comparison(multi_dirname,multi_rec['epoch'],
                bi_dirname,bi_rec['epoch'],
                ref_fname,
                slang,tlang,
                out_fname)

            print 'Finished: {} {} {}'.format(dataset,slang,tlang)

def run_comparison_onehot_phonetic(basedir,exp_conf_fname,out_dirname): 

    ## read the list of experiments to be analyzed 
    print 'Read list of experiments' 
    conf_df=pd.read_csv(exp_conf_fname,header=0,sep=',')

    print ' Comparing onehot vs phonetic experiments ' 
    datasets=[]

    print 'Getting to work ' 
    for dataset in datasets: 

        print '** Experiments for dataset: *** ' + dataset 

        ## phonetic
        phonetic_df=conf_df.query('dataset=="{dataset}" & exp=="2_multilingual" & representation=="phonetic"'.format(dataset=dataset))
        ## onehot
        onehot_df=conf_df.query('dataset=="{dataset}" & exp=="2_multilingual" & representation=="onehot_shared"'.format(dataset=dataset))

        for phonetic_rec in  [ x[1] for x in phonetic_df.iterrows() ]: 

            slang,tlang=(phonetic_rec['src'],phonetic_rec['tgt'])

            onehot_rec=onehot_df.query('src=="{}" & tgt=="{}"'.format(slang,tlang))
            onehot_rec=onehot_rec.iterrows().next()[1]

            phonetic_dirname = '{basedir}/results/sup/{dataset}/2_multilingual/phonetic/{edir}'.format(
                    basedir=basedir,dataset=dataset,edir=get_edir(phonetic_rec))
            onehot_dirname = '{basedir}/results/sup/{dataset}/2_multilingual/onehot_shared/{edir}'.format(
                    basedir=basedir,dataset=dataset,edir=get_edir(onehot_rec))

            ref_fname = '{basedir}/data/sup/mosesformat/{dataset}/{slang}-{tlang}/test.{tlang}'.format(
                basedir=basedir,dataset=dataset,slang=slang,tlang=tlang)

            out_fname = '{}/{}-{}-{}.png'.format(out_dirname,dataset,slang,tlang)


            print 'Starting: {} {} {}'.format(dataset,slang,tlang)

            transliteration_comparison(phonetic_dirname,phonetic_rec['epoch'],
                onehot_dirname,onehot_rec['epoch'],
                ref_fname,
                slang,tlang,
                out_fname)

            print 'Finished: {} {} {}'.format(dataset,slang,tlang)

def run_generate_analysis(basedir,exp_conf_fname): 
    """
     Run experiments to generate analysis files 
    """

    #def should_do_exp(rec): 
    #    """
    #    Filtering the list of experiments: this needs to be configured while running experiments 
    #    """
    #
    #    exp_check = rec['exp'] in ['2_multilingual','2_bilingual'] 
    #
    #    ## krishna
    #    dataset_check = rec['dataset'] in ['ar-slavic_latin', 'news_2015_reversed'] 
    #
    #    #### balaram 
    #    ##dataset_check = rec['dataset'] in ['slavic_latin-ar', 'news_2015_indic', 'news_2015_official' ] 
    #    #dataset_check = rec['dataset'] in ['news_2015_indic', 'news_2015_official' ] 
    #
    #    return exp_check and dataset_check 

    #def should_do_exp(rec): 
    #    """
    #     Always do experiments 
    #    """
    #    return True 

    def should_do_exp(rec): 
        """
         Never do experiments 
        """
        return False 
    


    ## read the list of experiments to be analyzed 
    print 'Read list of experiments' 
    conf_df=pd.read_csv(exp_conf_fname,header=0,sep=',')

    
    for rec in filter( should_do_exp, [x[1] for x in conf_df.iterrows()]): 

        edir=get_edir(rec)

        exp_dirname = '{basedir}/results/sup/{dataset}/{exp}/{rep}/{edir}'.format(
                basedir=basedir,dataset=rec['dataset'],rep=rec['representation'],exp=rec['exp'],edir=edir)

        ref_fname = '{basedir}/data/sup/mosesformat/{dataset}/{slang}-{tlang}/test.{tlang}'.format(
                basedir=basedir,dataset=rec['dataset'],slang=rec['src'],tlang=rec['tgt'])

        print 'Starting Experiment: ' + exp_dirname
        transliteration_analysis(exp_dirname,rec['epoch'],ref_fname,rec['src'],rec['tgt'])
        print 'Finished Experiment: ' + exp_dirname
        sys.stdout.flush()

def run_gather_metrics(basedir,exp_conf_fname,aug_exp_conf_name):
    """
     Run experiments to generate metrics for the experiments 
     It generates these error rates for experiments for which analysis has already been done, 
     and generates an new exp_conf_name with columns for the new experiments 
    """

    ## read the list of experiments to be analyzed 
    print 'Read list of experiments' 
    conf_df=pd.read_csv(exp_conf_fname,header=0,sep=',')
    
    augmented_data=[]

    for rec in [x[1] for x in conf_df.iterrows()]: 

        slang=rec['src']
        tlang=rec['tgt']
        epoch=rec['epoch']

        edir=get_edir(rec)

        exp_dirname = '{basedir}/results/sup/{dataset}/{exp}/{rep}/{edir}'.format(
                basedir=basedir,dataset=rec['dataset'],rep=rec['representation'],exp=rec['exp'],edir=edir)

        #  001test.nbest.bn-en.en
        eval_fname='{exp_dirname}/outputs/{epoch:03d}test.nbest.{slang}-{tlang}.{tlang}.eval'.format(
            exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)

        print 'Starting Experiment: ' + eval_fname

        if os.path.isfile(eval_fname):
            with codecs.open(eval_fname,'r','utf-8') as evalfile: 
        
                scores=[ float(l.strip().split(':')[1].strip())  for l in evalfile.readlines()] 
                rec['acc']=scores[0]
                rec['mf1']=scores[1]
                rec['mrr']=scores[2]
                rec['map']=scores[3]
                rec['a10']=scores[4]

        augmented_data.append(rec)
        print 'Finished Experiment: ' + exp_dirname
        print 
        sys.stdout.flush()

    new_df=pd.DataFrame(augmented_data)
    new_df.to_csv(aug_exp_conf_name,columns=list(conf_df.columns)+['mf1','mrr','map','a10'],index=False)

def run_lang_ind_err_rates(basedir,exp_conf_fname,aug_exp_conf_name): 
    """
     Run experiments to generate language independent error rates 
     It generates these error rates for experiments for which analysis has already been done, 
     and generates an new exp_conf_name with columns for the new experiments 
    """

    ## read the list of experiments to be analyzed 
    print 'Read list of experiments' 
    conf_df=pd.read_csv(exp_conf_fname,header=0,sep=',')
    
    augmented_data=[]

    for rec in [x[1] for x in conf_df.iterrows()]: 

        slang=rec['src']
        tlang=rec['tgt']
        epoch=rec['epoch']

        edir=get_edir(rec)

        exp_dirname = '{basedir}/results/sup/{dataset}/{exp}/{rep}/{edir}'.format(
                basedir=basedir,dataset=rec['dataset'],rep=rec['representation'],exp=rec['exp'],edir=edir)

        out_dirname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}'.format(
            exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)

        print 'Starting Experiment: ' + exp_dirname
        if os.path.isdir(out_dirname): 
            a_df=align.read_align_count_file('{}/alignment_count.csv'.format(out_dirname))
            rec['char_erate']=align.char_error_rate(a_df) 
            rec['ins_erate']=align.ins_error_rate(a_df) 
            rec['del_erate']=align.del_error_rate(a_df) 
            rec['sub_erate']=align.sub_error_rate(a_df)
        
        augmented_data.append(rec)
        print 'Finished Experiment: ' + exp_dirname
        print 
        sys.stdout.flush()

    new_df=pd.DataFrame(augmented_data)
    new_df.to_csv(aug_exp_conf_name,columns=list(conf_df.columns)+['char_erate','ins_erate','del_erate','sub_erate'],index=False)

if __name__ == '__main__': 
  
    basedir='/home/development/anoop/experiments/multilingual_unsup_xlit'
    exp_list='results_with_accuracy.csv'

    ## command to generate the analysis files for each experiment 
    #run_generate_analysis(basedir,exp_list) 

    ## command to compare bilingual and multilingual experiments 
    ## mkdir -p $basedir/analysis/bi_vs_multi/heat_maps 
    #run_comparison_bi_multi(basedir,exp_list,'{}/analysis/bi_vs_multi/heat_maps'.format(basedir)) 


    ## command to compare bilingual and multilingual experiments 
    ## /home/development/anoop/experiments/multilingual_unsup_xlit/analysis/onehot_vs_phonetic/heat_maps/
    #run_comparison_onehot_phonetic(basedir,exp_list,'{}/analysis/onehot_vs_phonetic/heat_maps'.format(basedir)) 

    # get language independent error rates 
    aug_exp_list='results_with_accuracy_new.csv'
    run_gather_metrics(basedir,exp_list,aug_exp_list)

    # get language independent error rates 
    exp_list='results_with_accuracy_new.csv'
    aug_exp_list='results_with_accuracy_new2.csv'
    run_lang_ind_err_rates(basedir,exp_list,aug_exp_list)

    #transliteration_analysis(
    #        '/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_indic/2_multilingual/onehot_shared/indic-indic',
    #        11,
    #        '/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/mosesformat/news_2015_indic/hi-bn/test.bn',
    #        'hi',
    #        'bn',
    #        )
        
    #transliteration_comparison(
    #        '/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/2_multilingual/onehot_shared/indic-en',
    #        37,
    #        '/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_reversed/2_bilingual/onehot/hi-en',
    #        22,
    #        '/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/mosesformat/news_2015_reversed/hi-en/test.en',
    #        'hi',
    #        'en',
    #        'out.png'
    #        )

