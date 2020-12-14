import pickle
import math
import pandas as pd
import numpy as np
from numpy import array

# Neural Net Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

# Neural Net Training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from pickle import load
import re
import sklearn.utils 
import math
import 


def regex_to_pattern_objects(regex_list):
    #regex_list = array of strings to be interpreted as regex
    pattern_objs = []
    for regex in regex_list:
        pattern_objs.append(re.compile(regex)) 
    return pattern_objs 

def remove_regex(tweet, *bad_patterns):
    #tweet = string
    #bad_patterns = a list of pattern objects to remove
    for pattern in bad_patterns:
        tweet = re.sub(pattern, "", tweet)
    return tweet
        
def to_lowercase(tweet): #is this necessary lol
    return tweet.lower()

regex_list = ['\n', 'RT', '&amp', '&#\d*;', '@\S*:', '@\S*', '!+', '"+', 'https?:\/\/t\.co\/\w*', '#', '&\S*;']
pattern_list = regex_to_pattern_objects(regex_list)