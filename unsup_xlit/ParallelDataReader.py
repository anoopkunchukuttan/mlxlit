import numpy as np
class ParallelDataReader():
	# Initializer and data reader
	def __init__(self, lang1, lang2, filename1, filename2, mapping, max_sequence_length):
		# Taking args and storing as class vars
		self.lang1 = lang1
		self.lang2 = lang2
		self.mapping = mapping
		self.c2i = dict()
		self.c2i[lang1] = mapping.get_c2i(lang1)
		self.c2i[lang2] = mapping.get_c2i(lang2)
		self.max_sequence_length = max_sequence_length

		self.sequences = dict()
		self.masks = dict()
		self.lengths = dict()
		self.num_words = dict()

		# Calling read_file for both the languages
		self.read_file(filename1, lang1)
		self.read_file(filename2, lang2)

		# For parallel data, number of words must be same for both datasets
		assert self.num_words[lang1] == self.num_words[lang2], filename1+" and "+filename2+" are of unequal lengths"
		self.num_words = self.num_words[lang1]

		# Vars for batch handling
		self._epochs_completed = 0
		self._current_index = 0
		
		#Shuffling data
		perm = list(np.arange(self.num_words))
		for lang in [self.lang1,self.lang2]:
			self.sequences[lang] = self.sequences[lang][perm]
			self.masks[lang] = self.masks[lang][perm]
			self.lengths[lang] = self.lengths[lang][perm]

	# Reads file with given name and lang
	def read_file(self, filename, lang):
		# Reading the file
		file_read = map(lambda x: ['GO']+(x.strip().split(' '))[:self.max_sequence_length-2]+['EOW'],open(filename,'r').readlines())

		#lengths is list of lengths of all words. i.e. a list of integers with size num_words in dataset 
		self.lengths[lang] = np.array(map(lambda x: len(x), file_read))
		self.num_words[lang] = len(self.lengths[lang])

		#Adding PAD to the words to make each word length = max_sequence_length
		file_read = map(lambda x: x+['PAD']*(self.max_sequence_length-len(x)),file_read)

		# replacing character with corresponding character id
		self.sequences[lang] = np.array([[self.c2i[lang][char] for char in word] for word in file_read], dtype = np.int32)

		# Creating masks. Mask has size = size of list of sequence. 
		# Corresponding to each PAD character there is a zero, for all other there is a 1
		self.masks[lang] = np.zeros([self.num_words[lang], self.max_sequence_length],dtype = np.float32)
		for i in range(self.num_words[lang]):
			self.masks[lang][i][:self.lengths[lang][i]]=1

	def get_next_batch(self,batch_size):
		# If epoch was completed in last call, reset current index
		if(self._current_index >= self.num_words):
			self._current_index = 0

		start = self._current_index
		end = min(start + batch_size, self.num_words)

		self._current_index = end

		batch_sequences1 = self.sequences[self.lang1][start:end]
		batch_masks1 = self.masks[self.lang1][start:end]
		batch_lengths1 = self.lengths[self.lang1][start:end]

		batch_sequences2 = self.sequences[self.lang2][start:end]
		batch_masks2 = self.masks[self.lang2][start:end]
		batch_lengths2 = self.lengths[self.lang2][start:end]

		if(self._current_index >= self.num_words):
			self._epochs_completed += 1

			# Shuffling as epoch is complete
			perm = list(np.arange(self.num_words))
			for lang in [self.lang1,self.lang2]:
				self.sequences[lang] = self.sequences[lang][perm]
				self.masks[lang] = self.masks[lang][perm]
				self.lengths[lang] = self.lengths[lang][perm]

		return batch_sequences1, batch_masks1, batch_lengths1, batch_sequences2, batch_masks2, batch_lengths2

	# Return entire data
	def get_data(self):
		return self.sequences[self.lang1], self.masks[self.lang1], self.lengths[self.lang1], self.sequences[self.lang2], self.masks[self.lang2], self.lengths[self.lang2]