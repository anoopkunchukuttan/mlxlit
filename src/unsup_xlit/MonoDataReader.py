import numpy as np

class MonoDataReader():
    # Initializer and data reader
    def __init__(self, lang, filename, mapping, max_sequence_length):
        # Taking args and storing as class vars
        self.lang = lang
        self.mapping = mapping
        self.c2i = mapping.get_c2i(lang)
        self.max_sequence_length = max_sequence_length

        # Reading the file
        # NOTE: -2 should be -1 
        file_read = map(lambda x: ['GO']+(x.strip().split(' '))[:max_sequence_length-2]+['EOW'],open(filename,'r').readlines())
        for s in file_read: 
            print ' '.join(s)

        #lengths is list of lengths of all words. i.e. a list of integers with size num_words in dataset 
        self.lengths = np.array(map(lambda x: len(x), file_read))
        self.num_words = len(self.lengths)

        #Adding PAD to the words to make each word length = max_sequence_length
        file_read = map(lambda x: x+['PAD']*(max_sequence_length-len(x)),file_read)

        # replacing character with corresponding character id
        self.sequences = np.array([[self.c2i[char] for char in word] for word in file_read], dtype = np.int32)

        # Creating masks. Mask has size = size of list of sequence. 
        # Corresponding to each PAD character there is a zero, for all other there is a 1
        self.masks = np.zeros([self.num_words, self.max_sequence_length],dtype = np.float32)
        for i in range(self.num_words):
            self.masks[i][:self.lengths[i]]=1

        # Vars to be used while returning batch_size batches
        self._epochs_completed = 0
        self._current_index = 0
        
        # Shuffling data
        perm = list(np.arange(self.num_words))
        self.sequences = self.sequences[perm]
        self.masks = self.masks[perm]
        self.lengths = self.lengths[perm]

    # Returns next batch of data with given batch size
    def get_next_batch(self,batch_size):
        # If epoch was completed in last call, reset current index
        if(self._current_index >= self.num_words):
            self._current_index = 0

        start = self._current_index
        end = min(start + batch_size, self.num_words)

        self._current_index = end

        batch_sequences = self.sequences[start:end]
        batch_masks = self.masks[start:end]
        batch_lengths = self.lengths[start:end]

        if(self._current_index >= self.num_words):
            self._epochs_completed += 1

            # Shuffling as epoch is complete
            perm = list(np.arange(self.num_words))
            self.sequences = self.sequences[perm]
            self.masks = self.masks[perm]
            self.lengths = self.lengths[perm]

        return batch_sequences, batch_masks, batch_lengths

    # Return entire dataset
    def get_data(self):
        return self.sequences, self.masks, self.lengths
