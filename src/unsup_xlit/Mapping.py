from collections import defaultdict
import numpy as np 

from indicnlp.script import  indic_scripts as isc
from indicnlp import langinfo as li

class Mapping():

    GO=u'GO'
    EOW=u'EOW'
    PAD=u'PAD'
    UNK=u'UNK'

    def finalize_vocab(self): 
        """
        Call after all vocabulary has been added via get_index
        """
        pass 

    def get_index(self,c,lang=None): 
        pass 

    def get_char(self,index,lang): 
        pass 

    def get_langs(self): 
        pass 

    def get_vocab_size(self): 
        pass 

    def get_vocab(self,lang): 
        pass 

    def get_bitvector_embedding_size(self,representation): 
        pass 

    def get_bitvector_embeddings(self,lang,representation): 
        pass 

    # Given sequence of character ids, return word.
    # A word is space separated character with GO, EOW (End of Word) and PAD character to make total length = max_sequence_length
    def get_word_from_ids(self,sequence,lang):
            return u' '.join([self.get_char(char_id,lang) for char_id in sequence])
    
    # Given a list of (sequence of character_ids), return list of words     
    def get_words_from_id_lists(self,sequences,lang):
            return [self.get_word_from_ids(sequence, lang) for sequence in sequences]

class IndicPhoneticMapping(Mapping):

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

    def get_bitvector_embedding_size(self,representation='phonetic'): 
        if representation=='phonetic':
            return isc.PHONETIC_VECTOR_LENGTH+len(self.addvocab_c2i)
        elif representation=='onehot': 
            return self.get_vocab_size()
        elif representation=='onehot_and_phonetic':
            return self.get_vocab_size()+isc.PHONETIC_VECTOR_LENGTH  ### FIXME: this size is wrong

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
        phv=self.get_phonetic_bitvector_embeddings(lang)[:,:isc.PHONETIC_VECTOR_LENGTH]
        ohv=np.identity(self.get_vocab_size())
        return np.concatenate([ohv,phv],1)

    def get_bitvector_embeddings(self,lang,representation='phonetic'): 
    
        if representation=='phonetic':
            return self.get_phonetic_bitvector_embeddings(lang)
        elif representation=='onehot': 
            return self.get_onehot_bitvector_embeddings(lang)
        elif representation=='onehot_and_phonetic': 
            return self.get_onehot_phonetic_bitvector_embeddings(lang)

class IndicPhonetic2Mapping(Mapping):

    def __init__(self): 

        ### members
        self.vocab_c2i=defaultdict(lambda: len(self.vocab_c2i))
        self.vocab_i2c={}
        self.indic_i2pid={}
        self.lang_list=set()
        
        ## state members 
        self.update_mode=True

        ## add standard vocabulary 
        self.vocab_c2i[Mapping.GO]
        self.vocab_c2i[Mapping.EOW]
        self.vocab_c2i[Mapping.PAD]
        self.vocab_c2i[Mapping.UNK]

    def finalize_vocab(self): 
        """
        Call after all vocabulary has been added via get_index
        """
        for c,i in self.vocab_c2i.iteritems(): 
            self.vocab_i2c[i]=c
        self.update_mode=False

    def get_index(self,c,lang=None): 

        if len(c)==1 and isc.in_coordinated_range(c,lang): 
            pid=isc.get_offset(c,lang)
            c_hi=isc.offset_to_char(pid,'hi')
            if (not self.update_mode) and (c_hi not in self.vocab_c2i): 
                c_hi=Mapping.UNK
            index=self.vocab_c2i[c_hi]
            self.indic_i2pid[index]=pid
        else:
            if (not self.update_mode) and (c not in self.vocab_c2i): 
                c=Mapping.UNK
            index=self.vocab_c2i[c]

        if self.update_mode: 
            self.lang_list.add(lang)

        return index

    def get_char(self,index,lang=None): 

        c=None 

        if index in self.indic_i2pid: 
            pid=self.indic_i2pid[index]
            c=isc.offset_to_char(pid,lang)
        else: 
            c=self.vocab_i2c.get(index,Mapping.UNK)

        return c
    
    def get_langs(self): 
        return self.lang_list 

    def get_vocab_size(self): 
        return len(self.vocab_c2i)

    def get_vocab(self,lang=None): 
        return self.vocab_c2i.keys()

    def get_bitvector_embedding_size(self,representation='phonetic'): 
        if representation=='onehot': 
            return self.get_vocab_size()
        elif representation=='phonetic': 
            num_non_phonetic_chars=self.get_vocab_size()-len(self.indic_i2pid)
            return isc.PHONETIC_VECTOR_LENGTH+num_non_phonetic_chars    
        elif representation=='onehot_and_phonetic': 
            return self.get_vocab_size()+isc.PHONETIC_VECTOR_LENGTH

    def get_phonetic_bitvector_embeddings(self,lang):
        """
        Create bit-vector embeddings for vocabulary items. For phonetic chars,
        use phonetic embeddings, else use 1-hot embeddings
        """

        non_phonetic_chars=filter(lambda x: x not in self.indic_i2pid, range(self.get_vocab_size()))
        bitvector_embedding=np.zeros((self.get_vocab_size(), isc.PHONETIC_VECTOR_LENGTH+len(non_phonetic_chars)))

        for p,index in enumerate(non_phonetic_chars): 
            bitvector_embedding[index,isc.PHONETIC_VECTOR_LENGTH+p]=1

        for index in self.indic_i2pid.keys(): 
            bitvector_embedding[index,:isc.PHONETIC_VECTOR_LENGTH]=isc.get_phonetic_feature_vector(self.get_char(index,lang),lang)
        
        return bitvector_embedding

    def get_onehot_bitvector_embeddings(self,lang): 
        return np.identity(self.get_vocab_size())
    
    def get_onehot_phonetic_bitvector_embeddings(self,lang): 
        """
        Concatenation of the one-hot and phonetic represenation
        For vocabulary items which don't have a phonetic representation, they are represented in the one-hot
        component. So, the phonetic component does not have the onehot subcomponent to avoid onehot duplication
        (unlike the call in get_phonetic_bitvector_embeddings)
        """
        phv=self.get_phonetic_bitvector_embeddings(lang)[:,:isc.PHONETIC_VECTOR_LENGTH]
        ohv=np.identity(self.get_vocab_size())
        return np.concatenate([ohv,phv],1)

    def get_bitvector_embeddings(self,lang,representation='phonetic'): 
    
        if representation=='phonetic':
            return self.get_phonetic_bitvector_embeddings(lang)
        elif representation=='onehot': 
            return self.get_onehot_bitvector_embeddings(lang)
        elif representation=='onehot_and_phonetic': 
            return self.get_onehot_phonetic_bitvector_embeddings(lang)

class CharacterMapping(Mapping):

    def __init__(self): 

        ### members
        self.vocab_c2i=defaultdict(lambda: len(self.vocab_c2i))
        self.vocab_i2c={}
        self.lang_list=set()
        
        ## state members 
        self.update_mode=True

        ## add standard vocabulary 
        self.vocab_c2i[Mapping.GO]
        self.vocab_c2i[Mapping.EOW]
        self.vocab_c2i[Mapping.PAD]
        self.vocab_c2i[Mapping.UNK]

    def finalize_vocab(self): 
        """
        Call after all vocabulary has been added via get_index
        """
        for c,i in self.vocab_c2i.iteritems(): 
            self.vocab_i2c[i]=c
        self.update_mode=False

    def get_index(self,c,lang=None): 
        if (not self.update_mode) and (c not in self.vocab_c2i): 
            c=Mapping.UNK
        index=self.vocab_c2i[c]

        if self.update_mode: 
            self.lang_list.add(lang)

        return index

    def get_char(self,index,lang=None): 
        return self.vocab_i2c.get(index,Mapping.UNK)

    def get_langs(self): 
        return self.lang_list 

    def get_vocab_size(self): 
        return len(self.vocab_c2i)

    def get_vocab(self,lang=None): 
        return self.vocab_c2i.keys()

    def get_bitvector_embedding_size(self,representation='onehot'): 
        if representation=='onehot': 
            return self.get_vocab_size()
        else: 
            ##TODO: throw exception
            pass 

    def get_bitvector_embeddings(self,lang,representation='onehot'): 
    
        if representation=='onehot': 
            return np.identity(self.get_vocab_size())
        else: 
            ##TODO: throw exception
            pass 

