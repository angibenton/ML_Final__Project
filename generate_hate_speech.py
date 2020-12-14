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


model = Sequential([
    Embedding(num_words+1, 50, input_length=train_len),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(100, activation='relu'),
    Dense(num_words-2, activation='softmax')
])

def generate_hate(model,text,length):

    # Tokenize the input string
    passing_tokens = tokens.texts_to_sequences([text])
    length = length+len(passing_tokens[0])

    while len(passing_tokens[0]) < length:
    
        padded_sentence = pad_sequences(passing_tokens[-19:],maxlen=19)
        op = model.predict(np.asarray(padded_sentence).reshape(1,-1))
        passing_tokens[0].append(op.argmax()+1)
        
    return " ".join(map(lambda x : reverse_word_map[x],passing_tokens[0]))
    

    
    
 ### Reference: 'Simple Text Generation' https://towardsdatascience.com/simple-text-generation-d1c93f43f340
