import collections
import os,sys

import numpy as np

import codecs 

from indicnlp import langinfo as li
from indicnlp.normalize import indic_normalize
from indicnlp.script import indic_scripts as isc

def convert_to_unk(infname,outfname,lang): 
    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(outfname,'w','utf-8') as outfile:
        
        for i,line in enumerate(iter(infile)):
            print  i, len(line.strip().split())

            chars=[ c if isc.in_coordinated_range(c,lang) else u'UNK' for c in line.strip().split()]
            outfile.write(u' '.join(chars)+u'\n')


def find_unk_words(infname,outfname,lang): 
    with codecs.open(infname,'r','utf-8') as infile, \
         codecs.open(outfname,'w','utf-8') as outfile:
        
        for i,line in enumerate(iter(infile)):
            print  i, len(line.strip().split()),
            #chars=[ isc.in_coordinated_range(c,lang) for c in line.strip().split()]
            #if False in chars: 
            #outfile.write(line.strip() + u'======' + u' '.join([str(c) for c in chars]) + u'\n')
            x=u' ' + line.strip()
            print(x.encode('utf-8'))
                #outfile.write(line.strip() + u'======' + u' '.join([u'{:x}'.format(ord(c)) for c in line.strip().split()]) + u'\n')

def normalize_corpus_file(infname,outfname,lang):

    factory=indic_normalize.IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang)

    # DO normalization 
    with codecs.open(infname,'r','utf-8') as ifile:
        with codecs.open(outfname,'w','utf-8') as ofile:
            for line in ifile.readlines():
                normalized_line=normalizer.normalize(line)
                normalized_line=u' '.join([ c if len(c)==1 else u' '.join(c)  for c in normalized_line.strip().split()])
                ofile.write(normalized_line+u'\n')

if __name__ == '__main__':
    #find_unk_words(*sys.argv[1:])
    normalize_corpus_file(*sys.argv[1:])
