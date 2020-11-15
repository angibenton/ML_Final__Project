import re
import pandas as pd



#PREPROCESSING STEP 1: CLEANING THE TWEET STRINGS 

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

#TODO: make the regex list exactly what we want
regex_list = ['RT', '&amp', '&#\d*;', '@\S*:', '@\S*', '!+', '"+', 'https?:\/\/t\.co\/\w*', '#']
pattern_list = regex_to_pattern_objects(regex_list)

tweets = pd.read_csv("raw_data.csv", index_col = 0)
tweets["tweet"] = tweets["tweet"].apply(remove_regex, args = (pattern_list))
tweets["tweet"] = tweets["tweet"].apply(to_lowercase)


#OTHER PREPROCESSING:
# - seperate out the classes -> two options
# - balance the classes 
# - train & test split 
# - output format - .csv 2 columns: class, string 

tweets.to_csv("preprocessed_data.csv")



