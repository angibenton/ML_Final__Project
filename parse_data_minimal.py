import pandas as pd
tweets = pd.read_csv("raw_data.csv", index_col = 0)

good_tweets = tweets[tweets["class"] == 2]
offensive_tweets = tweets[tweets["class"] == 1]
hate_tweets = tweets[tweets["class"] == 0]

print(hate_tweets)
print(offensive_tweets)
print(good_tweets)

print("TOTAL TWEETS: ", len(tweets))
print("GOOD TWEETS: ", len(good_tweets))
print("OFFENSIVE TWEETS: ", len(offensive_tweets))
print("HATEFUL TWEETS: ", len(hate_tweets))
