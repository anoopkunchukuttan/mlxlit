import itertools as it, operator 
import codecs 
import sys,os
from collections import defaultdict

import numpy as np
from scipy.misc.common import logsumexp
import itertools as it
import random

def read_monolingual_corpus(corpus_fname): 
    with codecs.open(corpus_fname,'r','utf-8') as infile:
        for w in infile: 
            yield w.strip().split()

def write_monolingual_corpus(corpus_fname,output_list): 
    with codecs.open(corpus_fname,'w','utf-8') as outfile:
        for output in output_list: 
            outfile.write(u' '.join(output) + '\n')

def convert_output_format(infname,outfname): 

    write_monolingual_corpus( outfname, 
        it.imap(lambda chars: u' '.join(it.takewhile(lambda x:x != u'EOW',it.dropwhile(lambda x:x==u'GO',chars))) , 
                read_monolingual_corpus(infname))
        )

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

def early_stop_best(metric,maxepoch,*options):
    
    loss_scores=None

    if metric=='loss':
        loss_scores=read_validloss_from_log(maxepoch,*options)
    elif metric=='accuracy':                 
        loss_scores=read_negacc(maxepoch,*options)

    min_epoch=min(enumerate(loss_scores),key=operator.itemgetter(1))
    return min_epoch[0]+1

def early_stop_best_multilingual(metric,maxepoch,*options):
    
    loss_scores=None

    if metric=='loss':
        loss_scores=read_validloss_from_log(maxepoch,*options)
    elif metric=='accuracy':                 
        loss_scores=read_negacc_multilingual(maxepoch,*options)

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

if __name__=='__main__': 

    commands = {
            'convert_output_format': convert_output_format,
            'transfer_pivot_translate': transfer_pivot_translate,
            'compute_accuracy': compute_accuracy,
            'compute_accuracy_multilingual': compute_accuracy_multilingual,
    }

    commands[sys.argv[1]](*sys.argv[2:])

