import numpy as np
import os
import nltk
import pickle
import random
import pandas as pd
import re
from collections import defaultdict
from bs4 import BeautifulSoup

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

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

def group_articles_by_topic(dataset):
    data = defaultdict(list)

    for text, topics in dataset:
        if len(topics) == 0:
            continue
        for topic in topics:
            data[topic].append(text)

    return data

def clean_text(file):
    regex_list = ['RT', '&amp', '&#\d*;', '@\S*:', '@\S*', '!+', '"+', 'https?:\/\/t\.co\/\w*', '#', '&\S*;']
    pattern_list = regex_to_pattern_objects(regex_list)
    tweets = pd.read_csv("raw_data.csv", index_col = 0)
    tweets["tweet"] = tweets["tweet"].apply(remove_regex, args = (pattern_list))
    tweets["tweet"] = tweets["tweet"].apply(to_lowercase)

def load_data(mode, datadir, num_articles, problem):
    """
    Load data.

    Args:
        mode: train or test
        datadir: directory containing SGML files 
    
    Returns:
        List of documents and list of topic labels
    """
    if mode == 'train':
        if(problem == 'offensive-normal'):
            tweets = pd.read_csv(os.path.join(datadir,"p1_train.csv"))
        else:
            tweets = pd.read_csv(os.path.join(datadir,"p2_train.csv"))
    else:
        if(problem == 'offensive-normal'):
            tweets = pd.read_csv(os.path.join(datadir,"p1_test.csv"))
        else:
            tweets = pd.read_csv(os.path.join(datadir,"p2_test.csv"))

    X = tweets["tweet"]
    Y = tweets["class"]

    X = X.values.tolist()
    Y = Y.values.tolist()
    return X,Y
