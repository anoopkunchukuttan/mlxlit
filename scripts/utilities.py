import itertools as it, operator 
import codecs 
import sys,os,glob
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.misc import logsumexp
import itertools as it
import random

from indicnlp.script import  indic_scripts as isc
from indicnlp import loader

import buckwalter

def read_monolingual_corpus(corpus_fname): 
    with codecs.open(corpus_fname,'r','utf-8') as infile:
        for w in infile: 
            yield w.strip().split()

def write_monolingual_corpus(corpus_fname,output_list): 
    with codecs.open(corpus_fname,'w','utf-8') as outfile:
        for output in output_list: 
            outfile.write(u' '.join(output) + '\n')

def read_parallel_corpus(src_fname,tgt_fname):
    with codecs.open(src_fname,'r','utf-8') as src_file,\
         codecs.open(tgt_fname,'r','utf-8') as tgt_file:

        for sline, tline in it.izip(iter(src_file),iter(tgt_file)):           
            yield ( sline.strip().split() , tline.strip().split() )

def convert_output_format(infname,outfname): 

    write_monolingual_corpus( outfname, 
        it.imap(lambda chars: u' '.join(it.takewhile(lambda x:x != u'EOW',it.dropwhile(lambda x:x==u'GO',chars))) , 
                read_monolingual_corpus(infname))
        )

def find_best_lm_weight(dirname,slang,tlang): 

    loss_scores=[]

    for fname in glob.glob('{}/*.eval'.format(dirname)):
        with codecs.open(fname,'r') as resfile:
            acc=float(resfile.readline().strip().split(':')[1].strip())
            param=float(fname.split('/')[-1].split('_')[0])
            loss_scores.append((param,-acc))

    best_param,negacc=min(loss_scores,key=operator.itemgetter(1))
    print '{}|{}'.format(best_param,-negacc),
    return (best_param,-negacc)

def read_negacc(maxepoch,dirname,slang,tlang): 

    loss_scores=[]

    for epoch in range(1,maxepoch+1):
        with codecs.open('{dirname}/{epoch:03d}test.nbest.{slang}-{tlang}.{tlang}.eval'.format(
                    dirname=dirname,slang=slang,tlang=tlang,epoch=epoch),'r') as resfile:
            acc=float(resfile.readline().strip().split(':')[1].strip())
            loss_scores.append(-acc)

    return loss_scores

def read_acc(maxepoch,dirname,slang,tlang): 

    return [ -x for x in read_negacc(maxepoch,dirname,slang,tlang) ]

def find_lang_pairs(dirname): 

    lang_pairs=set()
    for fname in filter(lambda x: x.find('.eval')>=0, os.listdir(dirname)): 
        lp=tuple(fname.split('/')[-1].split('.')[2].split('-'))
        lang_pairs.add(lp)

    return lang_pairs 

def read_negacc_multilingual(maxepoch,dirname,scale_values=False): 

    ## find the list of all language pairs 
    lang_pairs=find_lang_pairs(dirname)

    loss_scores=defaultdict(list)

    for slang,tlang in lang_pairs: 
        for epoch in range(1,maxepoch+1):
            with codecs.open('{dirname}/{epoch:03d}test.nbest.{slang}-{tlang}.{tlang}.eval'.format(
                        dirname=dirname,slang=slang,tlang=tlang,epoch=epoch),'r') as resfile:
                acc=float(resfile.readline().strip().split(':')[1].strip())
                loss_scores[tuple([slang,tlang])].append(-acc)

    if scale_values: 
        for slang,tlang in lang_pairs: 
            loss_s_t = loss_scores[tuple([slang,tlang])]
            max_val  = max(loss_s_t)
            min_val  = min(loss_s_t)
            loss_scores[tuple([slang,tlang])] = map(lambda x: (x-max_val)/(max_val-min_val), loss_s_t )

    loss_scores_lists=loss_scores.values()
    ave_loss_scores=map( lambda x:sum(x)/len(x), zip(*loss_scores_lists) )

    return ave_loss_scores

def read_validloss_from_log(maxepoch,log_fname,slang=None,tlang=None):

    valid_loss=[]
    with codecs.open(log_fname,'r','utf-8') as log_file: 
        for line in log_file: 
            if line.find('Epochs Completed')>=0 and \
               line.find('Validation loss')>=0:   
                score=float(line.split(':')[-1].strip())
                epoch_n=line.split(':')[1].replace(
                    'Validation loss','').strip()
                valid_loss.append(score)
                                   
    return valid_loss[:min(maxepoch,len(valid_loss))]

def early_stop_best_multilingual(metric,maxepoch,*options):
    
    loss_scores=None

    if metric=='loss':
        loss_scores=read_validloss_from_log(maxepoch,*options)
    elif metric=='accuracy':                 
        loss_scores=read_negacc_multilingual(maxepoch,*options)

    min_epoch=min(enumerate(loss_scores),key=operator.itemgetter(1))
    return min_epoch[0]+1

def early_stop_best_str(metric,maxepoch,*options):
    
    loss_scores=None

    if metric=='loss':
        loss_scores=read_validloss_from_log(maxepoch,*options)
    elif metric=='accuracy':                 
        loss_scores=read_negacc(maxepoch,*options)

    min_epoch=min(enumerate(loss_scores),key=operator.itemgetter(1))

    print '{}|{}'.format(min_epoch[0]+1,(-1.0 if (metric=='accuracy') else 1.0)*min_epoch[1]),

def early_stop_best(metric,maxepoch,*options):
    
    loss_scores=None

    if metric=='loss':
        loss_scores=read_validloss_from_log(maxepoch,*options)
    elif metric=='accuracy':                 
        loss_scores=read_negacc(maxepoch,*options)

    min_epoch=min(enumerate(loss_scores),key=operator.itemgetter(1))

    return min_epoch[0]+1

def compute_accuracy(exp_dirname,slang,tlang,maxepoch): 

    maxepoch=int(maxepoch)
    
    min_epoch=early_stop_best('accuracy',maxepoch,'{}/{}'.format(exp_dirname,'validation'),slang,tlang)
    #min_epoch=early_stop_best('loss',maxepoch,'{}/{}'.format(exp_dirname,'train.log'),slang,tlang)
    accuracies=read_acc(maxepoch,'{}/{}'.format(exp_dirname,'outputs'),slang,tlang) 
    #print '{}'.format(min_epoch)
    print '{}|{}'.format(min_epoch, accuracies[min_epoch-1])

def compute_accuracy_multilingual(exp_dirname,maxepoch): 

    maxepoch=int(maxepoch)
    min_epoch=early_stop_best_multilingual('accuracy',maxepoch,'{}/{}'.format(exp_dirname,'validation'))
    print '{}'.format(min_epoch)

    lang_pairs=find_lang_pairs('{}/{}'.format(exp_dirname,'outputs'))
    for slang,tlang in lang_pairs: 
        accuracies=read_acc(maxepoch,'{}/{}'.format(exp_dirname,'outputs'),slang,tlang) 
        print '{}|{}|{}|{}'.format(slang,tlang,min_epoch, accuracies[min_epoch-1])

#def early_stop_patience(log_fname, patience_str):
#
#    valid_loss=read_validloss_from_log(log_fname)
#    patience=int(patience_str)
#
#    c_pos=0
#    c_min=10000
#
#    while c_pos < len(valid_loss): 
#        p, v = min(     enumerate(valid_loss[  c_pos :   min(c_pos+patience,len(valid_loss))  ]), 
#                        key= operator.itemgetter(1)
#                  )
#        p=c_pos+p
#
#        if v>=c_min: 
#            break
#        else: 
#            c_pos=p
#            c_min=v
#
#    print c_pos+1

### Methods for parsing n-best lists
def parse_nbest_line(line):
    """
        line in n-best file 
        return list of fields
    """
    fields=[ x.strip() for x in  line.strip().split('|||') ]
    fields[0]=int(fields[0])
    fields[3]=float(fields[3])
    return fields

def iterate_nbest_list(nbest_fname): 
    """
        nbest_fname: moses format nbest file name 
        return iterator over tuple of sent_no, list of n-best candidates

    """

    infile=codecs.open(nbest_fname,'r','utf-8')
    
    for sent_no, lines in it.groupby(iter(infile),key=lambda x:parse_nbest_line(x)[0]):
        parsed_lines = [ parse_nbest_line(line) for line in lines ]
        yield((sent_no,parsed_lines))

    infile.close()

def transfer_pivot_translate(output_s_b_fname,output_b_t_fname,output_final_fname,n=10): 

    b_t_iter=iter(iterate_nbest_list(output_b_t_fname))

    with codecs.open(output_final_fname,'w','utf-8') as output_final_file: 
        for (sent_no, parsed_bridge_lines) in iterate_nbest_list(output_s_b_fname):     
            candidate_list=[]
            for parsed_bridge_line in parsed_bridge_lines: 
                (_,parsed_tgt_lines)=b_t_iter.next()
                for parsed_tgt_line in parsed_tgt_lines:
                    output=parsed_tgt_line[1]
                    score=parsed_bridge_line[3]+parsed_tgt_line[3]
                    candidate_list.append((output,score))

            ## if there are duplicates their log probabilities need to be summed 
            candidate_list.sort(key=lambda x:x[0])
            group_iterator=it.groupby(candidate_list,key=lambda x:x[0])
            candidate_list=[ (k,logsumexp([x[1] for x in group]))  for k, group in group_iterator ]
                
            candidate_list.sort(key=lambda x:x[1],reverse=True)

            for c,score in candidate_list[:n]:
                output_final_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( sent_no, c, '0.0 0.0 0.0 0.0', score  ) )

def read_best_epoch(exp_conf_fname,dataset,exp,representation,src,tgt): 

    conf_df=pd.read_csv(exp_conf_fname,header=0,sep=',')
    query_str='dataset=="{}" & exp=="{}" & representation=="{}" & src=="{}" and tgt=="{}"'.format(
                dataset,exp,representation,src,tgt)
    resp_df=conf_df.query(query_str)
    if resp_df.shape[0]==0:
        print 'Invalid Experiment' 
        sys.exit(1)
    elif resp_df.shape[0]>1:
        print 'More than one result. Please be more specific in the query' 
        sys.exit(1)
    elif resp_df.shape[0]==1:
        print resp_df.iterrows().next()[1]['epoch']


def shuffle(infname,outfname): 

    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(outfname,'w','utf-8') as outfile:

        data_list=list(iter(infile))
        random.shuffle(data_list)

        for line in data_list: 
            outfile.write(line)

def lcsr_indic(srcw,tgtw,slang,tlang):
    score_mat=np.zeros((len(srcw)+1,len(tgtw)+1))

    for si,sc in enumerate(srcw,1): 
        for ti,tc in enumerate(tgtw,1): 
            so=isc.get_offset(sc,slang)
            to=isc.get_offset(tc,tlang)

            if isc.in_coordinated_range_offset(so) and isc.in_coordinated_range_offset(to) and so==to: 
                score_mat[si,ti]=score_mat[si-1,ti-1]+1.0
            elif not (isc.in_coordinated_range_offset(so) or isc.in_coordinated_range_offset(to)) and sc==tc: 
                score_mat[si,ti]=score_mat[si-1,ti-1]+1.0
            else: 
                score_mat[si,ti]= max(
                    score_mat[si,ti-1],
                    score_mat[si-1,ti])

    return (score_mat[-1,-1]/float(max(len(srcw),len(tgtw))),float(len(srcw)),float(len(tgtw)))

def lcsr_any(srcw,tgtw,slang,tlang):
    score_mat=np.zeros((len(srcw)+1,len(tgtw)+1))

    for si,sc in enumerate(srcw,1): 
        for ti,tc in enumerate(tgtw,1): 

            if sc==tc: 
                score_mat[si,ti]=score_mat[si-1,ti-1]+1.0
            else: 
                score_mat[si,ti]= max(
                    score_mat[si,ti-1],
                    score_mat[si-1,ti])

    return (score_mat[-1,-1]/float(max(len(srcw),len(tgtw))),float(len(srcw)),float(len(tgtw)))

def lcsr(srcw,tgtw,slang,tlang):

    if slang==tlang or not isc.is_supported_language(slang) or not isc.is_supported_language(tlang):
        return lcsr_any(srcw,tgtw,slang,tlang)
    else:  
        return lcsr_indic(srcw,tgtw,slang,tlang)

#def orthographic_similarity(src_fname,tgt_fname,out_fname,src_lang,tgt_lang):
#
#    total=0.0
#    n=0.0
#    with codecs.open(out_fname,'w','utf-8') as out_file: 
#        for sline, tline in read_parallel_corpus(src_fname,tgt_fname):           
#            score,sl,tl=lcsr(sline,tline,src_lang,tgt_lang)
#            total+=score
#            n+=1.0
#
#            out_file.write(u'{}|{}|{}\n'.format(score,sl,tl))
#
#        print '{}-{}" {}'.format(src,tgt,total/n)

def orthographic_similarity(src_fname,tgt_fname,src_lang,tgt_lang):

    total=0.0
    n=0.0

    for sline, tline in read_parallel_corpus(src_fname,tgt_fname):           
        score,sl,tl=lcsr(sline,tline,src_lang,tgt_lang)
        total+=score
        n+=1.0

    print '{}-{} {}'.format(src_lang,tgt_lang,total/n)

def extract_common_corpus_wikidata(data_dir, p_lang, c0_lang, c1_lang, outdir): 

    data_cache=defaultdict(lambda : [set(),set()])

    # read corpus 0
    with codecs.open('{}/{}-{}.txt'.format(data_dir,p_lang,c0_lang),'r','utf-8') as c_file: 
        for line in c_file: 
            en_l, c_l = line.strip().split(u'|')
            data_cache[en_l][0].add(c_l)

    # read corpus 1                
    with codecs.open('{}/{}-{}.txt'.format(data_dir,p_lang,c1_lang),'r','utf-8') as c_file: 
        for line in c_file: 
            en_l, c_l = line.strip().split(u'|')
            data_cache[en_l][1].add(c_l)


    # write the common data

    # from language c0 to c1
    with codecs.open(outdir+'/train.{}'.format(c0_lang),'w','utf-8') as c0_outfname, \
         codecs.open(outdir+'/train.{}'.format(c1_lang),'w','utf-8') as c1_outfname:
        parallel_list=[]
        for en_l, other_l_lists in data_cache.iteritems(): 
            if len(other_l_lists[0]) >0 and len(other_l_lists[1]) >0 : 
                for c0_str in  other_l_lists[0] :  
                    for c1_str in  other_l_lists[1] :  
                        #if len(c0_str_w)>3:
                        c0_outfname.write(c0_str+u'\n')
                        c1_outfname.write(c1_str+u'\n')

def remove_terminal_halant(infname,outfname,lang): 
    """
    input and output are moses format files 
    """

    def process(w): 
        return w[:-1] if isc.is_halant(isc.get_phonetic_feature_vector(w[-1],lang)) else w

    with codecs.open(outfname,'w','utf-8') as outfile:
        for (sent_no, parsed_bridge_lines) in iterate_nbest_list(infname):     
            for parsed_bridge_line in parsed_bridge_lines:
                parsed_bridge_line[1] = process(parsed_bridge_line[1])
                outfile.write( u'{} ||| {} ||| {} ||| {}\n'.format( *parsed_bridge_line ) )  

def simple_ensemble(res1_fname,res2_fname,res_ens_fname): 
    """
        Ensemble the results from two systems, by just averaging the scores from the two systems
    """

    with codecs.open(res_ens_fname,'w','utf-8') as res_ens_file: 
        for ( (r1_sentno, r1_cand), (r2_sentno, r2_cand) ) in \
                it.izip( iterate_nbest_list(res1_fname), iterate_nbest_list(res2_fname) ): 
            
            #final_cands=set()                
            #final_cands.update([ x[1] for x in r1_cand ])
            #final_cands.update([ x[1] for x in r2_cand ]) 

            #for final_cand in final_cands: 
            #    res_ens_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( r1_sentno, final_cand, -1.0, -1.0 ) )  

            def make_unique_cands(cand_list): 
                """
                just consider the first candidate in each list of the ensemble 
                """
                unique_cand_list = []
                for x in cand_list: 
                    if x[1] not in [ y[0] for y in unique_cand_list ]: 
                        unique_cand_list.append((x[1],x[3]))
                return unique_cand_list  

            r1_unique_cand = make_unique_cands(r1_cand)
            r2_unique_cand = make_unique_cands(r2_cand)

            final_cands=defaultdict(list)

            for x in r1_unique_cand: 
                final_cands[x[0]].append(x[1])
            for x in r2_unique_cand: 
                final_cands[x[0]].append(x[1])

            ## average scores 
            sorted_scores=[]
            for cand,score_list in final_cands.iteritems(): 
                #sorted_scores.append( (cand,sum(score_list)/len(score_list)) )
                sorted_scores.append( (cand,sum(score_list)/2.0) )
        
            ## sort by score 
            sorted_scores.sort(key=lambda x:x[1],reverse=True)                

            ## write output score
            for cand,score in sorted_scores: 
                #print score
                res_ens_file.write( u'{} ||| {} ||| {} ||| {}\n'.format( r1_sentno, cand, -1.0, score ) )  

def xlit_detail_eval(infname,outfname,lang):
    """
        Added transliteration to the detailed evaluation file 
    """
    ### Create Arabic to Roman transliterator using buckwalter scheme
    xlitor=None
    if lang=='ar': 
        xlitor=buckwalter.Transliterator(mode='a2r')

    assert (xlitor is not None) 

    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(outfname,'w','utf-8') as outfile: 

        for n,line in enumerate(infile): 
            
            if n > 0:
                w=line.strip().replace(u'"',u'').split(u',')
                w.append(xlitor.transliterate(w[1]))
                w.append(xlitor.transliterate(w[5]))
                w.append(xlitor.transliterate(w[8]))
                outfile.write( u','.join(w) + u'\n' )
            else: 
                outfile.write(line.strip() + u',"First candidate (xlit)","Best matching reference (xlit)","References (clit)"' + u'\n'   )

def to_python_literal(infname,outfname): 
    with codecs.open(infname,'r','utf-8') as infile,\
         codecs.open(outfname,'w','utf-8') as outfile: 
        
        outfile.write('[\n')
        for line in infile: 
            c = line.strip()
            outfile.write(u'u"\\u{:04x}"'.format(ord(c)) +u',\n')
        outfile.write(']\n')

if __name__=='__main__': 

    loader.load()

    commands = {
            'convert_output_format': convert_output_format,

            'transfer_pivot_translate': transfer_pivot_translate,
            'compute_accuracy': compute_accuracy,
            'compute_accuracy_multilingual': compute_accuracy_multilingual,
            'read_best_epoch': read_best_epoch,
            'early_stop_best': early_stop_best_str,

            'shuffle': shuffle,

            'find_best_lm_weight': find_best_lm_weight,

            'orthographic_similarity': orthographic_similarity,
            'extract_common_corpus_wikidata': extract_common_corpus_wikidata,

            'remove_terminal_halant': remove_terminal_halant,

            'simple_ensemble': simple_ensemble,

            'to_python_literal': to_python_literal,

            'xlit_detail_eval': xlit_detail_eval,
    }

    commands[sys.argv[1]](*sys.argv[2:])

