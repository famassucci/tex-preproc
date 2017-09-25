# -*- coding: utf-8 -*-
import pandas as pd
import nltk, pkg_resources, string, os, re, pickle, collections
from nltk.corpus import conll2000, stopwords

def nounDict():
    '''
        A function that returns 'n'
        by default, to make a handy defaultdict
        when stemming words
    '''
    return 'n'

## a few 'standard' NLTK functions
## to learn chunks from POS tagging

def tags_since_dt(sentence, i):
     tags = set()
     for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
     return '+'.join(sorted(tags))

def npchunk_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
     else:
         nextword, nextpos = sentence[i+1]
     return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos,
             "prevpos+pos": "%s+%s" % (prevpos, pos),
             "pos+nextpos": "%s+%s" % (pos, nextpos),
             "tags-since-dt": tags_since_dt(sentence, i)}

class ConsecutiveNPChunkTagger(nltk.TaggerI): # [_consec-chunk-tagger]
    
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( # [_consec-use-maxent]
                                                      train_set, algorithm='megam', trace=0)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)
    
    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

## Here a class to deal with text corpora
class MyLanguageCorpus ():
    '''
    A class that allows to load a corpus of interest,
    and pre-process it in order to do nlp.
    '''
    def __init__ (self, stopWordPath = False, megamPath = False):
        '''
        Initialise the class.
        The method initialises
        a. self.texts = a list of texts contained in the corpus
        b. self.IDS : the id of each text
        c. self.stopWords : a list of any additional stopword
        d. self.Lemmatizer : a defaultdict to stem words. It returns 'n' for each token, except for verbs, for which it returns 'v'
        ---------------------------
        KeyWord arguments:
        i. stopWordPath : a file path to load extra stop words
        ii. megamPath : a path to the megam binary to train the chunker

        '''
        ## a container of the texts in the corpus
        self.texts = []
        ## input files
        self.in_files = []
        ## ids of the texts
        self.IDS = []
        ## stopwords
        self.stopWords = []
        if stopWordPath:
            ## a path to a text file containing a list of
            ## stop words
            self.stopWordPath = stopWordPath
            self.stopWords = [ l.strip() for l in open (self.stopWordPath).readlines() ]
        ## initialise the dict for stemming
        self.Lemmatizer = collections.defaultdict(nounDict)
        self.Lemmatizer ['v'] = 'v'
        
        
        ## try to load a trained chunker
        trainPath = pkg_resources.resource_filename('MyLanguageCorpus', '')
        if os.path.exists ( '%s/trainedChunker.pkl' %trainPath ):
    
            fin = open ( '%s/trainedChunker.pkl'%trainPath, 'rb' )
            self.chunker = pickle.load ( fin )
            fin.close()

        else:
            train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
            
            print ("Training chunker...")
            if 'MEGAM' in os.environ:
                MEGAM = os.environ ['MEGAM']
            elif megamPath:
                MEGAM = megamPath
                os.environ ['MEGAM'] = megamPath
                    
            chunker = ConsecutiveNPChunker(train_sents)
            fout = open ( '%s/trainedChunker.pkl' %trainPath, 'wb' )
            pickle.dump ( chunker, fout )
            fout.close()
            self.chunker = chunker
    
    def fullTokenize (self, text, removalPattern = False, gensim = False):
        
        
        '''
        A method to process texts by tokenising, stemming and retrieving noun phrases.
        The method
        a. removes useless stuff (stopwords and the like)
        b. stems words (removing plurals and verbal tenses)
        c. detects noun-phrases and joins them with hypens
        ---------------------------
        Mandatory Arguments:
        i. text : a string to be processed
        
        KeyWord Arguments:
        i. removalPattern : a regex pattern for sentences to be discarded
        ---------------------------
        Returns:
        finalText : a tokenized and stemmed list of the input text
        '''
        ## split the text into sentences
        if removalPattern:
            sentences = [ p for p in nltk.sent_tokenize(text) if ( not\
                ('ociety' in str(p) and ('©' in str(p) or 'ublished' in p) and len(p.split()) < 10) and \
                removalPattern.search( p + ' ') == None) ] # NLTK default sentence segmenter
        else:
            sentences = [ p for p in nltk.sent_tokenize(text) if ( not\
                                                                  ('ociety' in str(p) and ('©' in str(p) or 'ublished' in p) and len(p.split()) < 10) ) ] # NLTK default sentence segmenter
        sentences = [nltk.word_tokenize(sent) for sent in sentences] # NLTK word tokenizer
        sentences = [nltk.pos_tag(sent) for sent in sentences] # NLTK POS tagger

        # lemmatizing stuff
        sentences =  [[(nltk.WordNetLemmatizer().lemmatize(w, self.Lemmatizer[t[0].lower()]), t) for w,t in sent] for sent in sentences ]
        
        ## retain the nounphrases as a single-hypened word
        finalText = []
        for sentence in sentences:
            ## tag the sentence with the chunker
            result = self.chunker. tagger.tag(sentence)
            result = filter( lambda _ : _[0][1] != 'DT', result)

            ## merge the nounphrases
            phrase = []
            previous = 'O'
            for w0, t in result:
                if t == 'I-NP' and previous != 'O':
                    
                    phrase[-1] += '-%s' %w0[0]
                
                else:
                    phrase.append ( w0[0] )
                
                previous = t
            ## remove stopstuff
            filterStop = filter(lambda x: '-' in x and (len(set(x.split('-')) & set(stopwords.words('english'))) < len (x.split('-'))/2) or not ('-' in x), phrase)
            phrase = list (filterStop)
            finalText.extend (phrase)
    

        return finalText
    
    def guessParser (self, ftype, textcolumn, idcolumn, **kwargs):
        '''
            A method to guess the parser of the input files.
            Valid parsers are: csv, table, excel.
            
            Mandatory Arguments:
            i. ftype : the type of file to be parsed (csv, table, excel)
            ii. textcolumn : the column to be parsed as text
            iii. idcolumn : the column to be parsed as id
            ---------------------------
            Returns
            parser :  a pandas file parser
        '''
    
        ## get the parse corresponding to ftype
        ## and update the keyword args for id/text column
        if ftype  == 'csv':
            parser = pd.read_csv
            kwargs ['usecols'] = [textcolumn, idcolumn]
        elif ftype == 'table':
            parser = pd.read_table
            kwargs ['usecols'] = [textcolumn, idcolumn]
        elif ftype == 'excel':
            parser = pd.read_excel
            kwargs ['parse_cols'] = [textcolumn, idcolumn]
        
        ## return parser
        return parser
        
    def addTexts (self, filelist, ftype, textcolumn = None, idcolumn = None, **kwargs):
    
        '''
        A method that reads the corpus from a 
        list of files, passed through the list
        of strings 'filelist'.
        ---------------------------
        Mandatory arguments
        i. filelist : a list of file paths to be parsed
        ii. ftype : the type of files to be parsed. Assumed to be the same for all
        ---------------------------
        Keyword Arguments
        i. textcolumn : if reading a table, the column to use to read the text
        ii. idcolumn : if reading a table, the column to use as id
        '''
        ## loop over the files and read them
        for filename in filelist:
            self.readInput (filename, ftype, textcolumn, idcolumn, **kwargs)

    
    def readInput (self, fname, ftype, textcolumn, idcolumn, **kwargs):
    
        '''
        A method to read different kind of inputs and import them.
        It is possible to read plain text or tables in tsv csv and excel formats.
        ---------------------------
        Mandatory arguments
        i. fname : the filename to parse
        ii. ftype : the type of files to be parsed
        iii. textcolumn : the column to parse the text (if a table)
        iv. idcolumn : the column to parse as id (if a table)
        '''
    
        ## if reading a table
        if ftype in ['csv', 'table', 'excel']:
            ## guess the parser
            parser = self.guessParser(ftype, textcolumn, idcolumn, **kwargs)
            ## and add the text
            df = parser (fname,  dtype= object, index_col=False, **kwargs)
            self.texts.extend ( df [textcolumn].apply ( lambda _ : str(_).lower() ).values)
            self.IDS.extend ( df [idcolumn].values)
            del df
        ## if reading a plain text
        elif ftype == 'text':
            text_in = open ( fname , 'rb').read().decode ('utf-8').lower()
            ## append the text
            self.texts.append( text_in.lower() )
            ## extend the ids
            self.IDS.append (fname.split ('/')[-1].split ('.txt')[0])

        else:
            raiseTypeError ('Invalid file type specified.\nValid types are csv, table, excel and text')
    
    def preprocessTexts (self):
        '''
        A method that preprocesses all texts in the corpus.
        It loops over self.texts and does
        tokenization/stemming/nounphrase extraction.
        '''

        ## some handy patterns to remove
        ## via regex rules
        removal = ['all rights reserved.', '</inf>', '</sup>', 'nan', 'background:', 'objective:' , 'objectives:' ]
        sWords =  '|'.join(removal)
        pattern = re.compile("\\b(%s)\\W" %sWords, re.I)
        
        
        ## loop over texts
        for idx, txt in enumerate (self.texts):
            ## preprocess
            outText = self.fullTokenize ( txt , removalPattern = pattern)
            self.texts [ idx ] = outText
    

    def dumpTexts (self, outdir):
        '''
            A method that dumps the texts onto textfiles
            to be processed by some other software.
            ---------------------------
            Mandatory arguments:
            i. ourdir : the directory to put the texts.
            If not existent, it will create it.
        '''

        ## if the directory does not exist, create it
        if not os.path.exists (outdir):
            os.mkdir (outdir)

        for txt, fname in zip (self.texts, self.IDS):
            ## dump the text to a file identified by the text id
            outFname = fname
            outText = ' '.join(txt)
            outFile = open ("%s/%s.txt" %(outdir, outFname), "w")
            outFile.write (outText)


