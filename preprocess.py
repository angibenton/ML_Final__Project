import re
import pandas as pd
import sklearn.utils 



#cleaning the tweets

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
regex_list = ['RT', '&amp', '&#\d*;', '@\S*:', '@\S*', '!+', '"+', 'https?:\/\/t\.co\/\w*', '#', '&\S*;']
pattern_list = regex_to_pattern_objects(regex_list)

tweets = pd.read_csv("raw_data.csv", index_col = 0)
tweets["tweet"] = tweets["tweet"].apply(remove_regex, args = (pattern_list))
tweets["tweet"] = tweets["tweet"].apply(to_lowercase)

#strip everything but the 'tweet' and 'class' columns
tweets = tweets.drop(['count','hate_speech','offensive_language','neither'], axis = 1)

#get different subsets
tweets_hate = tweets[tweets['class'] == 0].copy(deep = True);
tweets_off = tweets[tweets['class'] == 1].copy(deep = True);
tweets_good = tweets[tweets['class'] == 2].copy(deep = True);
tweets_off_and_hate = pd.concat([tweets_off, tweets_hate])
tweets_good_and_off = pd.concat([tweets_off, tweets_good])

#class balancing
#downsample the union of offensive & hate to match good
tweets_off_and_hate = tweets_off_and_hate.sample(len(tweets_good))
#downsample the union of good & offensive to match hate
tweets_good_and_off = tweets_good_and_off.sample(len(tweets_hate))

#assign binary classes {-1, 1} for each of the two classification problems that we have
#problem 1
tweets_off_and_hate['class'] = -1
tweets_good['class'] = 1

print(tweets_off_and_hate)
print(tweets_good)

#problem 2
tweets_hate['class'] = -1
tweets_good_and_off['class'] = 1

print(tweets_good_and_off)
print(tweets_hate)

#TODO train and test split, concatenate classes together and then randomly shuffle 
#save as csv 

tweets.to_csv("preprocessed_data.csv")



