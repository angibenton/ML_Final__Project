import re
import pandas as pd
import sklearn.utils 
import math


# --- Clean up the tweet strings --- 
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

#TODO: modify the regex list if needed
regex_list = ['\n', 'RT', '&amp', '&#\d*;', '@\S*:', '@\S*', '!+', '"+', 'https?:\/\/t\.co\/\w*', '#', '&\S*;']
pattern_list = regex_to_pattern_objects(regex_list)

tweets = pd.read_csv("./data/raw_data.csv", index_col = 0)
tweets["tweet"] = tweets["tweet"].apply(remove_regex, args = (pattern_list))
tweets["tweet"] = tweets["tweet"].apply(to_lowercase)

# --- Strip all columns except for 'tweet' and 'class' ---
tweets = tweets.drop(['count','hate_speech','offensive_language','neither'], axis = 1)

# --- Isolate the 3 classes --- 
tweets_hate = tweets[tweets['class'] == 0].copy(deep = True);
tweets_off = tweets[tweets['class'] == 1].copy(deep = True);
tweets_good = tweets[tweets['class'] == 2].copy(deep = True);
# --- Isolate the compound classes  { offensive + hate } and  { offensive + good } --- 
tweets_off_and_hate = pd.concat([tweets_off, tweets_hate])
tweets_off_and_good = pd.concat([tweets_off, tweets_good])

# ---- Balance the classes --- 
#problem 1: downsample the union of { offensive + hate } to match  { good }
tweets_off_and_hate = tweets_off_and_hate.sample(len(tweets_good))
#problem 2: downsample the union of { good + offensive } to match  { hate }
tweets_off_and_good = tweets_off_and_good.sample(len(tweets_hate))

# --- Assign binary classes {-1, 1} for each problem --- 
#problem 1
tweets_off_and_hate['class'] = -1
tweets_good['class'] = 1
#problem 2
tweets_hate['class'] = -1
tweets_off_and_good['class'] = 1

# --- Train and test split --- 
TRAIN_FRAC = 0.8
#problem 1
total = len(tweets_good)
split_index = math.floor(total * TRAIN_FRAC)
p1_train = pd.concat([tweets_good[0 : split_index],tweets_off_and_hate[0 : split_index]])
p1_test = pd.concat([tweets_good[split_index : total], tweets_off_and_hate[split_index : total]])
#problem 2
total = len(tweets_hate)
split_index = math.floor(total * TRAIN_FRAC)
p2_train = pd.concat([tweets_hate[0 : split_index],tweets_off_and_good[0 : split_index]])
p2_test = pd.concat([tweets_hate[split_index : total], tweets_off_and_good[split_index : total]])

# -- Shuffle the final datasets --- 
p1_train = sklearn.utils.shuffle(p1_train)
p1_test = sklearn.utils.shuffle(p1_test)
p2_train = sklearn.utils.shuffle(p2_train)
p2_test = sklearn.utils.shuffle(p2_test)

# --- Save as csv --- 
p1_train.to_csv("data/p1_train.csv", index = False)
p1_test.to_csv("data/p1_test.csv", index = False)
p2_train.to_csv("data/p2_train.csv", index = False)
p2_test.to_csv("data/p2_test.csv", index = False)


