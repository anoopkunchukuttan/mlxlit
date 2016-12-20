import itertools as it 
import codecs 
import sys

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

if __name__=='__main__': 

    commands={
            'convert_output_format': convert_output_format,
    }

    commands[sys.argv[1]](*sys.argv[2:])

   
    
