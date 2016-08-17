from collections import defaultdict
import numpy as np 

########Importing indicnlp library for phonetic feature vector

# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME="indicnlp/indic_nlp_library"
# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES="indicnlp/indic_nlp_resources"
import sys
sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader
loader.load()
from indicnlp.script import  indic_scripts as isc

class Mapping():
	# c2i: character to id dictionaries. key is language code e.g. hi, kn etc. c2i[lang] is a defaultdict.
	# To get id for a character use: c2i[lang][char]. No need to add anything explicitly. Defaultdict will take care
	# i2c: id to character dictionaries. structure and usage same as c2i

	def __init__(self):
		self.c2i = dict()
		self.i2c = None

	# 'char_' is a character of language 'lang'
	# Structure of vector returned:
	# Size: 41
	# Bit: 0 to 37, phonetic features from indicnlp library of the character
	# Bit: 38, 39 and 40 are set for GO, EOW, PAD resp.
	def get_feature_vector(self, char_, lang):
		if(char_ == 'GO'):
			a=np.zeros([41])
			a[38]=1
			return a
		elif(char_ == 'EOW'):
			a=np.zeros([41])
			a[39]=1
			return a
		elif(char_ == 'PAD'):
			a=np.zeros([41])
			a[40]=1
			return a
		else:
			return np.append(isc.get_phonetic_feature_vector(char_.decode('utf-8'),lang),[0.,0.,0.])

	# return character to id dictionary of language lang
	# Must be called after all data is read
	def get_c2i(self,lang):
		if lang not in self.c2i:
			self.c2i[lang] = defaultdict(lambda: len(self.c2i[lang]))
		return self.c2i[lang]

	# Generates i2c dictionaries from c2i dictionaries. This should be called only after all data is read.
	# It is safer to call this after completely reading data, otherwise it is automatically called by other functions
	def generate_i2c(self):
		self.i2c = dict()
		for lang in self.c2i.keys():
			self.i2c[lang]=dict()
			for char in self.c2i[lang].keys():
				idx = self.c2i[lang][char]
				self.i2c[lang][idx]=char

	# Return id to character dictionary of the language 'lang'
	def get_i2c(self,lang):
		if(self.i2c == None):
			self.generate_i2c()
		return self.i2c[lang]

	# Generated and return return phonetic vectors for all languages
	# a dictionary is returned, language codes are whose keys
	# Value of phoenetic_vectors[lang] is np-array of dimensions (vocab_size of lang) x (feature vector size = 41)
	def get_phonetic_vectors(self):
		if(self.i2c == None):
			self.generate_i2c()
		phonetic_vectors = dict()
		for lang in self.c2i.keys():
			phonetic_vectors[lang] = np.asarray([self.get_feature_vector(self.i2c[lang][i],lang) for i in range(len(self.i2c[lang]))])
		return phonetic_vectors

	# Return list of language codes of all languages it knows
	def get_langs(self):
		return self.c2i.keys()

	# Return a dictionary {lang: (vocab_size of lang)....}
	def get_vocab_sizes(self):
		vocab_sizes = dict()
		for lang in self.c2i.keys():
			vocab_sizes[lang] = len(self.c2i[lang])
		return vocab_sizes

	# Given sequence of character ids, return word.
	# A word is space separated character with GO, EOW (End of Word) and PAD character to make total length = max_sequence_length
	def get_word_from_ids(self,sequence,lang):
		return ' '.join([self.i2c[lang][char_id] for char_id in sequence])


	# Given a list of (sequence of character_ids), return list of words	
	def get_words_from_id_lists(self,sequences,lang):
		return [self.get_word_from_ids(sequence, lang) for sequence in sequences]