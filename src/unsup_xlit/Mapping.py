from collections import defaultdict
import numpy as np 

from indicnlp.script import  indic_scripts as isc
from indicnlp import langinfo as li

class Mapping():

    GO=u'GO'
    EOW=u'EOW'
    PAD=u'PAD'
    UNK=u'UNK'

    def __init__(self): 

        ### members
        self.addvocab_c2i=defaultdict(lambda: li.COORDINATED_RANGE_END_INCLUSIVE + 1 + len(self.addvocab_c2i))
        self.addvocab_i2c={}
        self.lang_list=set()
        
        ## state members 
        self.update_mode=True

        ## add standard vocabulary 
        self.addvocab_c2i[Mapping.GO]
        self.addvocab_c2i[Mapping.EOW]
        self.addvocab_c2i[Mapping.PAD]
        self.addvocab_c2i[Mapping.UNK]

    def finalize_vocab(self): 
        """
        Call after all vocabulary has been added via get_index
        """
        for c,i in self.addvocab_c2i.iteritems(): 
            #print c.encode('utf-8')
            self.addvocab_i2c[i]=c
        self.update_mode=False

    def get_index(self,c,lang=None): 
        if len(c)==1 and isc.in_coordinated_range(c,lang): 
            index=isc.get_offset(c,lang)
        else:
            if (not self.update_mode) and (c not in self.addvocab_c2i): 
                c=Mapping.UNK
            index=self.addvocab_c2i[c]

        if self.update_mode: 
            self.lang_list.add(lang)

        return index

    def get_char(self,index,lang): 
        if isc.in_coordinated_range_offset(index): 
            c=isc.offset_to_char(index,lang)
        else: 
            c=self.addvocab_i2c.get(index,Mapping.UNK)
        return c

    def get_langs(self): 
        return self.lang_list 

    def get_vocab_size(self): 
        return (li.COORDINATED_RANGE_END_INCLUSIVE + 1 + len(self.addvocab_c2i))

    def get_vocab(self,lang): 
        return [ self.get_char(i,lang) for i in range(0,self.get_vocab_size()) ]

    def get_bitvector_embedding_size(self,representation): 
        if representation=='phonetic':
            return isc.PHONETIC_VECTOR_LENGTH+len(self.addvocab_c2i)
        elif representation=='onehot': 
            return self.get_vocab_size()
        elif representation=='onehot_and_phonetic':
            return self.get_vocab_size()+isc.PHONETIC_VECTOR_LENGTH

    def get_phonetic_bitvector_embeddings(self,lang):
        """
        Create bit-vector embeddings for vocabulary items. For phonetic chars,
        use phonetic embeddings, else use 1-hot embeddings
        """
    
        ##  phonetic embeddings for basic characters 
        pv=isc.get_phonetic_info(lang)[1]
    
        ## vocab statistics
        pv=np.copy(pv)
        org_shape=pv.shape
        additional_vocab=self.get_vocab_size()-org_shape[0]
    
        ##  new rows added 
        new_rows=np.zeros([additional_vocab,pv.shape[1]])
        pv=np.concatenate([pv,new_rows])
    
        ##  new columns added 
        new_cols=np.zeros([pv.shape[0],additional_vocab])
        pv=np.concatenate([pv,new_cols],axis=1)
    
        assert( (pv.shape[0]-org_shape[0]) == (pv.shape[1]-org_shape[1]) )
    
        ## 1-hot embeddings for new characters 
        for j,k in zip(range(org_shape[0],pv.shape[0]),range(org_shape[1],pv.shape[1])): 
            pv[j,k]=1
    
        return pv
    
    def get_onehot_bitvector_embeddings(self,lang): 
        return np.identity(self.get_vocab_size())
    
    def get_onehot_phonetic_bitvector_embeddings(self,lang): 
        """
        Concatenation of the one-hot and phonetic represenation
        For vocabulary items which don't have a phonetic representation, they are represented in the one-hot
        component. So, the phonetic component does not have the onehot subcomponent to avoid onehot duplication
        (unlike the call in get_phonetic_bitvector_embeddings)
        """
        ## TODO: alternative representation - can be deleted later
        #return np.concatenate([self.get_onehot_bitvector_embeddings(lang),self.get_phonetic_bitvector_embeddings(lang)],1)

        #### create the onehot component of the representation 
        ov=np.identity(self.get_vocab_size())

        #### create the phonetic componet of the representation 

        ##  phonetic embeddings for basic characters 
        pv=isc.get_phonetic_info(lang)[1]
    
        ## vocab statistics
        pv=np.copy(pv)
        org_shape=pv.shape
        additional_vocab=self.get_vocab_size()-org_shape[0]
    
        ##  new rows added 
        new_rows=np.zeros([additional_vocab,pv.shape[1]])
        pv=np.concatenate([pv,new_rows])
    
        return np.concatenate([ov,pv],1)

    def get_bitvector_embeddings(self,lang,representation): 
    
        if representation=='phonetic':
            return self.get_phonetic_bitvector_embeddings(lang)
        elif representation=='onehot': 
            return self.get_onehot_bitvector_embeddings(lang)
        elif representation=='onehot_and_phonetic': 
            return self.get_onehot_phonetic_bitvector_embeddings(lang)

    # Given sequence of character ids, return word.
    # A word is space separated character with GO, EOW (End of Word) and PAD character to make total length = max_sequence_length
    def get_word_from_ids(self,sequence,lang):
            return u' '.join([self.get_char(char_id,lang) for char_id in sequence])
    
    # Given a list of (sequence of character_ids), return list of words     
    def get_words_from_id_lists(self,sequences,lang):
            return [self.get_word_from_ids(sequence, lang) for sequence in sequences]

