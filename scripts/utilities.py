import itertools as it 
import codecs 
import sys

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

    commands={
            'convert_output_format': convert_output_format,
            'transfer_pivot_translate': transfer_pivot_translate,
    }

    commands[sys.argv[1]](*sys.argv[2:])

   
    
