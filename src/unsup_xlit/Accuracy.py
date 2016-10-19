import sys

# Finding actual end of words and trims the actual target sequences
# starts from character after GO and take till EOW

target_sequences = map(lambda x: x.split()+['EOW'], open(sys.argv[1],'r').readlines())
# target_sequences = [ x[ 1 : x.index('EOW') ] for x in target_sequences]
lengths = map(len,target_sequences)             # Actual sequence length
num_words = len(target_sequences)
num_chars = sum(lengths)+0.0            # Total number of character

# Trim predicted sequences so as to match size with the actual words
predicted_sequences = map(lambda x: x.split(), open(sys.argv[2],'r').readlines())
predicted_sequences_trimmed = [predicted_sequences[i][1:lengths[i]+1] for i in range(num_words)]

correct_words = 0.0
correct_chars = 0.0

# Iterating and increment correct_word/correct_chars when word/character match resp.
for j in range(num_words):
        if(target_sequences[j] == predicted_sequences_trimmed[j]):
                correct_words += 1
        for k in range(lengths[j]):
                if(target_sequences[j][k] == predicted_sequences_trimmed[j][k]):
                        correct_chars += 1

# Returns (word_accuracy,character accuracy tuple)
print "Word accuracy: {0}".format(correct_words/num_words)
print "Char accuracy: {0}".format(correct_chars/num_chars)
