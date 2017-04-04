from collections import defaultdict
import numpy as np 
import json

from indicnlp.script import  indic_scripts as isc
from indicnlp import langinfo as li

def create_special_token(token): 
    return '@@{}@@'.format(token)

def prefix_sequence_with_token(sequences,sequence_masks,sequence_lengths,token,mapping):

        lang_vec=np.ones([sequences.shape[0],1])*mapping.get_index(create_special_token(token))
        sd=np.concatenate([lang_vec,sequences],axis=1)[:,:-1]
    
        mask_vec=np.ones([sequences.shape[0],1])
        sm=np.concatenate([mask_vec,sequence_masks],axis=1)[:,:-1]
    
        sl=sequence_lengths+np.ones(sequences.shape[0])
    
        return sd, sm, sl

def get_mapping_instance(mapping_class): 

    if mapping_class == 'CharacterMapping': 
        return CharacterMapping()
    elif mapping_class == 'IndicPhoneticMapping':
        return IndicPhoneticMapping()

class Mapping():

    GO=u'GO'
    EOW=u'EOW'
    PAD=u'PAD'
    UNK=u'UNK'

    def save_mapping(self,mapping_file): 
        pass 

    def load_mapping(self,mapping_file): 
        pass 

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

    def save_mapping(self,mapping_file): 
        dump_data={}
        dump_data['class']='IndicPhoneticMapping'
        dump_data['c2i']=self.vocab_c2i
        dump_data['i2c']=self.vocab_i2c
        dump_data['i2pid']=self.indic_i2pid
        dump_data['langs']=list(self.lang_list)
        json.dump(dump_data,mapping_file)

    def load_mapping(self,mapping_file): 
        dump_data=json.load(mapping_file)
        self.vocab_c2i=dump_data['c2i']

        self.vocab_i2c={}
        for i_str,c in dump_data['i2c'].iteritems(): 
            self.vocab_i2c[int(i_str)]=dump_data['i2c'][i_str]

        self.indic_i2pid={}
        for i_str,pid in dump_data['i2pid'].iteritems(): 
            self.indic_i2pid[int(i_str)]=dump_data['i2pid'][i_str]

        self.lang_list=set(dump_data['langs'])
        self.update_mode=False

    def finalize_vocab(self): 
        """
        Call after all vocabulary has been added via get_index
        """
        for c,i in self.vocab_c2i.iteritems(): 
            self.vocab_i2c[i]=c
        self.update_mode=False

    def get_index(self,c,lang=None): 

        if len(c)==1 and lang is not None and isc.in_coordinated_range(c,lang): 
            pid=isc.get_offset(c,lang)
            c_hi=isc.offset_to_char(pid,'hi')
            if (not self.update_mode) and (c_hi not in self.vocab_c2i): 
                c_hi=Mapping.UNK
            index=self.vocab_c2i[c_hi]
            if self.update_mode: 
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
        if representation=='phonetic': 
            num_non_phonetic_chars=self.get_vocab_size()-len(self.indic_i2pid)
            return isc.PHONETIC_VECTOR_LENGTH+num_non_phonetic_chars    
        elif representation  in ['onehot','onehot_shared']: 
            return self.get_vocab_size()
        elif representation=='onehot_and_phonetic': 
            return self.get_vocab_size()+isc.PHONETIC_VECTOR_LENGTH

    def get_phonetic_bitvector_embeddings(self,lang):
        """
        Create bit-vector embeddings for vocabulary items. For phonetic chars,
        use phonetic embeddings, else use 1-hot embeddings
        """

        #non_phonetic_chars=filter(lambda x: x not in self.indic_i2pid, range(self.get_vocab_size()))
        non_phonetic_chars=filter(lambda x: x not in self.indic_i2pid, self.vocab_i2c.keys())
        print 'non_phonetic_chars: {}'.format(len(non_phonetic_chars))
        bitvector_embedding=np.zeros((self.get_vocab_size(), isc.PHONETIC_VECTOR_LENGTH+len(non_phonetic_chars)))

        for p,index in enumerate(non_phonetic_chars): 
            bitvector_embedding[index,isc.PHONETIC_VECTOR_LENGTH+p]=1

        for index in self.indic_i2pid.keys(): 
            bitvector_embedding[index,:isc.PHONETIC_VECTOR_LENGTH]=isc.get_phonetic_feature_vector(self.get_char(index,lang),lang)
        
        print bitvector_embedding.shape
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
        elif representation  in ['onehot','onehot_shared']: 
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

    def save_mapping(self,mapping_file): 
        dump_data={}
        dump_data['class']='CharacterMapping'
        dump_data['c2i']=self.vocab_c2i
        dump_data['i2c']=self.vocab_i2c
        dump_data['langs']=list(self.lang_list)
        json.dump(dump_data,mapping_file)

    def load_mapping(self,mapping_file): 
        dump_data=json.load(mapping_file)
        self.vocab_c2i=dump_data['c2i']

        self.vocab_i2c={}
        for i_str,c in dump_data['i2c'].iteritems(): 
            self.vocab_i2c[int(i_str)]=dump_data['i2c'][i_str]

        self.lang_list=set(dump_data['langs'])
        self.update_mode=False

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
        if representation  in ['onehot','onehot_shared']: 
            return self.get_vocab_size()
        else: 
            ##TODO: throw exception
            pass 

    def get_bitvector_embeddings(self,lang,representation='onehot'): 
    
        if representation  in ['onehot','onehot_shared']: 
            return np.identity(self.get_vocab_size())
        else: 
            ##TODO: throw exception
            pass 

