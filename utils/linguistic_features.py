# import packages
import numpy as np
import pandas as pd
import wordsegment as ws



def enhance_metadata(metadata, features='all'):
    """ Append columns to metadata dataframe based on column 'words'
                
    """

    # available options
    ortographic_features = ['w_length','n_vowels','n_consonants']
    lexical_features = ['uni_freq', 'bi_freq', 'func_word','count']
    position_features = ['position','position_end','is_first_word','is_last_word']

    # make list of features
    if features == 'all': features = ortographic_features +lexical_features + position_features 

    # use ws clean to lower case
    words = [word.lower() for word in metadata['word'].values]

    # itereate features and fill metadata
    for feature in features:
        #   ORTOGRAPHIC   ##############################
        if feature == 'w_length': 
            metadata[feature] = w_length(words)
        if feature == 'n_consonants':
            metadata[feature] = n_consonants(words)
        if feature == 'n_vowels':
            metadata[feature] = n_vowels(words)

        #   LEXICAL  ###################################
        if feature == 'uni_freq':
            metadata[feature] = unigram(words)
        if feature == 'bi_freq':
            metadata[feature] = bigram(words)
        if feature == 'func_word':
            metadata[feature] = function_word(words)
        if feature == 'count':
            metadata[feature] = count(words)

        #  POSITION  ###################################
        if feature == 'position':
            metadata[feature] = position(words)
        if feature == 'position_end':
            metadata[feature] = position_end(words)
        if feature == 'is_first_word':
            metadata[feature] = first_word(words)
        if feature == 'is_last_word':
            metadata[feature] = last_word(words)

    return metadata

################################################################################
#
#   ORTOGRAPHIC
#
################################################################################

def w_length(words):

    return [len(word) for word in words]


def n_consonants(words):

    consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNOPQRSTVWXYZ')

    return [sum(1 for c in word if c in consonants) for word in words]

    
def n_vowels(words):

    vowels = set('aeiouAEIOU')

    return [sum(1 for v in word if v in vowels) for word in words]


################################################################################
#
#   LEXICAL
#
################################################################################

def unigram(words):

    ws.load()

    values = []

    for word in words:
        try: values += [np.log10(ws.UNIGRAMS[word])]
        except: values += [1.]

    return values


def bigram(words):

    ws.load()

    values = [1.]

    for word1, word2 in zip(words[:-1],words[1:]):
        try: values += [np.log10(ws.BIGRAMS[' '.join([word1,word2])])]
        except: values += [1.]

    return values


def function_word(words):

    # 277 function words
    function_words = [
'a','about','above','across','after','afterwards','again','against','all','almost','alone','along','already', 'also','although','always','am','among','amongst','amoungst','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as','at','be','became','because','been','before','beforehand','behind','being','below','beside','besides','between','beyond','both','but','by','can','cannot','could','dare','despite','did','do','does','done','down','during','each','eg','either','else','elsewhere','enough','etc','even','ever','every','everyone','everything','everywhere','except','few','first','for','former','formerly','from','further','furthermore','had','has','have','he','hence','her','here','hereabouts','hereafter','hereby','herein','hereinafter','heretofore','hereunder','hereupon','herewith','hers','herself','him','himself','his','how','however','i','ie','if','in','indeed','inside','instead','into','is','it','its','itself','last','latter','latterly','least','less','lot','lots','many','may','me','meanwhile','might','mine','more','moreover','most','mostly','much','must','my','myself','namely','near','need','neither','never','nevertheless','next','no','nobody','none','noone','nor','not','nothing','now','nowhere','of','off','often','oftentimes','on','once','one','only','onto','or','other','others','otherwise','ought','our','ours','ourselves','out','outside','over','per','perhaps','rather','re','same','second','several','shall','she','should','since','so','some','somehow','someone','something','sometime','sometimes','somewhat','somewhere','still','such','than','that','the','their','theirs','them','themselves','then','thence','there','thereabouts','thereafter','thereby','therefore','therein','thereof','thereon','thereupon','these','they','third','this','those','though','through','throughout','thru','thus','to','together','too','top','toward','towards','under','until','up','upon','us','used','very','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon','wherever','whether','which','while','whither','who','whoever','whole','whom','whose','why','whyever','will','with','within','without','would','yes','yet','you','your','yours','yourself','yourselves'
                        ]

    return [1.*(word in function_words) for word in words]


def count(words):
    """ Count the number of times that the word has occurred previously in the experiment
    """

    values = []
    
    # dictionary whose keys are words and values number of occurrences
    D = {}

    for word in words:
        # if word is already in dict add 1 to the count
        try : D[word] +=1
        # otherwise add entrye to dict
        except :  D[word] = 1

        values += [D[word]]

    return values


################################################################################
#
#   POSITION
#
################################################################################

def first_word(words):

    return [0] + [ (word!='<s>')*(words[i]=='<s>') for i, word in enumerate(words[1:])]

def last_word(words):

    return [(word!='<s>')*(words[i+1]=='<s>') for i, word in enumerate(words[:-1])] +[False]


def position(words):

    values = []

    for i in range(len(words)):
        j = 0
        while words[i-j]!='<s>': j+=1
        values += [j]

    return values


def position_end(words):

    values = []

    for i in range(len(words)):
        j = 0
        while words[i+j]!='<s>': j+=1
        values += [-j]

    return values







