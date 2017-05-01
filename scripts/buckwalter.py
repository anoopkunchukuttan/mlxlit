##################################################
# Buckwalter Transliteration for python          #
# Follows the general transliteration scheme     #
# except that it allows for multiple decisions   #
# around whether or not to include all types     #
# of characters and diacritics                   #
#                                                #
# Note that this is not XML safe, and may clash  #
# with some punctuation marks (')                #
#                                                #
# Code is provided "as is", without any          #
# warranties or guarantees of any kind, either   #
# expressed or implied.                          #
#                                                #
# Originally Authored by Kenton Murray           #
# Qatar Computing Research Institute             #
# Doha, Qatar, 2014                              #
#                                                #
# Modified by Anoop Kunchukuttan                 #
# Indian Institute of Technology Bombay          #
# India, 2017                                    #
#  - Python API provided, instead of the         #
#    original commanline tool                    #
#                                                #
##################################################

import codecs, sys

class Transliterator(object): 

    def __init__(self,
                        mode='a2r',
                        is_hamza=True,
                        is_madda=True,
                        is_t=True,
                        is_harakat=True,
                        is_tatweel=True,
                    ): 
        """
            Converts characters in Arabic free text to Buckwater transliteration scheme and vice versa 
    
            mode: 'a2r': Arabic to Roman
                  'r2a': Roman to Arabic
    
            is_t: Include Tar Marbuta as a letter. Default=True
            is_hamza: Include Hamzas as a letter. Default=True
            is_madda: Include Alefs with Madda on top as a separate letter (otherwise just Alef). Default=True
            is_harakat: Include diacritics as separate letters (otherwise stripped). Default=True
            is_tatweel: Include tatweel as an underscore. Default=True
        """
    
        self.abjad = {u"\u0627":'A',
        u"\u0628":'b', u"\u062A":'t', u"\u062B":'v', u"\u062C":'j',
        u"\u062D":'H', u"\u062E":'x', u"\u062F":'d', u"\u0630":'*', u"\u0631":'r',
        u"\u0632":'z', u"\u0633":'s', u"\u0634":'$', u"\u0635":'S', u"\u0636":'D',
        u"\u0637":'T', u"\u0638":'Z', u"\u0639":'E', u"\u063A":'g', u"\u0641":'f',
        u"\u0642":'q', u"\u0643":'k', u"\u0644":'l', u"\u0645":'m', u"\u0646":'n',
        u"\u0647":'h', u"\u0648":'w', u"\u0649":'y', u"\u064A":'y'}
        
        # Create the reverse
        self.alphabet = {}
        if mode=='r2a':     
          for key in self.abjad:
            self.alphabet[self.abjad[key]] = key
        
        # Tar Mabutta
        if is_t:
          self.abjad[u"\u0629"] = 'p'
        else:
          self.abjad[u"\u0629"] = 't' # Some map to Ha ... decide
        
        # Hamza
        if is_hamza:
          self.abjad[u"\u0621"] = '\''
          self.abjad[u"\u0623"] = '>'
          self.abjad[u"\u0625"] = '<'
          self.abjad[u"\u0624"] = '&'
          self.abjad[u"\u0626"] = '}'
          self.abjad[u"\u0654"] = '\'' # Hamza above
          self.abjad[u"\u0655"] = '\'' # Hamza below
        else:
          self.abjad[u"\u0621"] = ''
          self.abjad[u"\u0623"] = 'A'
          self.abjad[u"\u0625"] = 'A'
          self.abjad[u"\u0624"] = '' # I don't think that the wa is pronounced otherwise ...
          self.abjad[u"\u0626"] = '' # Decide ...
          self.abjad[u"\u0654"] = ''
          self.abjad[u"\u0655"] = ''
        
        # Alef with Madda on Top
        if is_madda:
          self.abjad[u"\u0622"] = '|'
        else:
          self.abjad[u"\u0622"] = 'A'
        
        # Vowels/Diacritics
        if is_harakat:
          self.abjad[u"\u064E"] = 'a'
          self.abjad[u"\u064F"] = 'u'
          self.abjad[u"\u0650"] = 'i'
          self.abjad[u"\u0651"] = '~'
          self.abjad[u"\u0652"] = 'o'
          self.abjad[u"\u064B"] = 'F'
          self.abjad[u"\u064C"] = 'N'
          self.abjad[u"\u064D"] = 'K'
        else:
          self.abjad[u"\u064E"] = ''
          self.abjad[u"\u064F"] = ''
          self.abjad[u"\u0650"] = ''
          self.abjad[u"\u0651"] = ''
          self.abjad[u"\u0652"] = ''
          self.abjad[u"\u064B"] = ''
          self.abjad[u"\u064C"] = ''
          self.abjad[u"\u064D"] = ''
        
        # Tatweel
        if is_tatweel:
          self.abjad[u"\u0640"] = '_'
        else:
          self.abjad[u"\u0640"] = '' 
        
        ## Make sure mapping is right
        #for key in self.abjad:
        #  print key,
        #  print " ",
        #  print self.abjad[key]
        
        if mode=='r2a':     
          self.alphabet['|'] = u"\u0622"
          self.alphabet['a'] = u"\u064E"
          self.alphabet['u'] = u"\u064F"
          self.alphabet['i'] = u"\u0650"
          self.alphabet['~'] = u"\u0651"
          self.alphabet['o'] = u"\u0652"
          self.alphabet['F'] = u"\u064B"
          self.alphabet['N'] = u"\u064C"
          self.alphabet['K'] = u"\u064D"
          self.alphabet['\''] = u"\u0621"
          self.alphabet['>'] = u"\u0623"
          self.alphabet['<'] = u"\u0625"
          self.alphabet['&'] = u"\u0624"
          self.alphabet['}'] = u"\u0626"
          self.alphabet['p'] = u"\u0629"


        ## Save parameters
        self.mode=mode
        self.is_hamza=is_hamza
        self.is_madda=is_madda
        self.is_t=is_t
        self.is_harakat=is_harakat
        self.is_tatweel=is_tatweel

    def transliterate(self,inword):
        """
        transliterates the given text 
        
        inword: input text
        """
        if self.mode=='a2r': 
            return ''.join([ self.abjad[c] if (c in self.abjad) else c  for c in inword ])         
        # Take Buckwalter Transliterated Text and put it in vernacular
        elif self.mode=='r2a':     
          return u''.join([ self.alphabet[c] if (c in self.alphabet) else c  for c in inword ])         

if __name__ == '__main__':

    bw_xlit=Transliterator(mode=sys.argv[3])

    with codecs.open(sys.argv[1],'r','utf-8') as infile, \
         codecs.open(sys.argv[2],'w','utf-8') as outfile: 

        for line in infile:
            outfile.write(bw_xlit.transliterate(line))
