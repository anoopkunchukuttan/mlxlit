import numpy as np
import pandas as pd 

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

def plot_confusion_matrix(confusion_mat_fname,tlang,image_fname):
    """
    Plots a heat map of the confusion matrix of character alignments. 
    
    confusion_mat_fname: The input is a confusion matrix generated by the align script (csv format)
    tgt: target language of the transliteration
    
    For Indic scripts, the heat map shows characters in Devanagiri irrespective of target language for readability
        - Needs 'Lohit Devanagari' font to be installed 
    """
    
    #matplotlib.rc('font', family='Lohit Kannada') 
    matplotlib.rcParams.update({'font.size': 8})

    confusion_df=pd.read_pickle(confusion_mat_fname)

    schar=list(confusion_df.index)
    tchar=list(confusion_df.columns)
    i=0
    for c in schar: 
        if c in tchar: 
            confusion_df.ix[c,c]=0.0
    
    data=confusion_df.as_matrix()
    
    # # normalize along row
    # sums=np.sum(data,axis=1)
    # data=data.T/sums
    # data=data.T
    
    # # normalize along column
    # sums=np.sum(data,axis=0)
    # data=data/sums
    
    #s=np.sum(data)
    #data=data/s
    
    columns=list(confusion_df.columns)
    col_names=[ get_column_name(x,tlang) for x in columns]

    rows=list(confusion_df.index)
    row_names=[ get_column_name(x,tlang) for x in rows]
    
    plt.figure(figsize=(20,10))

    #plt.pcolor(data,cmap=plt.cm.gray_r,edgecolors='k')
    plt.pcolor(data,cmap=plt.cm.hot_r,edgecolors='k')
    
    #plt.pcolor(data,edgecolors='k')
    plt.colorbar()
    plt.xticks(np.arange(0,len(col_names))+0.5,col_names,rotation='vertical')
    plt.yticks(np.arange(0,len(row_names))+0.5,row_names)
    plt.xlabel('system')
    plt.ylabel('reference')
    
    plt.savefig(image_fname)
    plt.close()

def transliteration_analysis(exp_dirname,epoch,ref_fname,slang,tlang):
    """
    """
   
    ### generate file names 
    #out_fname='{exp_dirname}/outputs/{epoch:03d}test.nbest.{slang}-{tlang}.{tlang}'.format(
    #    exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)
    #out1b_fname='{exp_dirname}/outputs/{epoch:03d}test.1best.{slang}-{tlang}.{tlang}'.format(
    #    exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)

    ### save the output 
    #nwutil.convert_to_1best_format(out_fname,out1b_fname)
    #out_dirname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}'.format(
    #    exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang)
    #align.save_analysis_artifacts(ref_fname, out1b_fname, tlang, out_dirname)

    ## plot the confusion matrix  
    confmat_fname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}/confusion_mat.pickle'.format( 
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang) 
    confmat_img_fname='{exp_dirname}/outputs/{epoch:03d}_analysis_{slang}-{tlang}/confusion_mat.png'.format( 
        exp_dirname=exp_dirname,epoch=epoch,slang=slang,tlang=tlang) 
    plot_confusion_matrix(confmat_fname,tlang,confmat_img_fname)

if __name__ == '__main__': 

    transliteration_analysis(
            '/home/development/anoop/experiments/multilingual_unsup_xlit/results/sup/news_2015_indic/2_multilingual/onehot_shared/indic-indic',
            32,
            '/home/development/anoop/experiments/multilingual_unsup_xlit/data/sup/mosesformat/news_2015_indic/bn-hi/test.hi',
            'bn',
            'hi',
            )
        