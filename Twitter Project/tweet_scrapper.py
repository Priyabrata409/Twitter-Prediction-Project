#Data Scraping
#In this Session we will scrap data from twitter using Twitter APIwith the help of Python's Tweepy Library
import pandas as pd
import tweepy as tp
import os
import numpy as np
import nltk
#To access the Twitter API, you will need 4 things from your twitter App page. These keys are located in your Twitter app settings in the Keys and Access Token tab
consumer_key="apBoFDyHs2uwg1VLT81WoPTMA"
consumer_secret="TazlJnEcRuoLxCQGPddxzjE94VcOoxww1pCtCy1ys9zEpbAnUG"
access_token="1333996570170126337-UG3vemuxePuEQ0rs0fQSZTviDU3Hii"
access_secret="Z7WfCuQZ6RSrGM0juEwaKF2WfOYd0qmQ6Bi8dMUcw6jq6"
#Note:-For security I have not mentioned the keys here
auth=tp.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
api=tp.API(auth,wait_on_rate_limit=True)

#Search Tweet for Keyword Hyderabad
new_search="Hyderabad"+" -filter:retweets"## We are not considering Retweets
tweets=tp.Cursor(api.search,q=new_search,lang="en").items(700)
data=[[tweet.user.id,tweet.user.name,tweet.user.screen_name,tweet.text,tweet.user.location,tweet.place,"Hyderabad"] for tweet in tweets]
# tweet.user.id="Tweet user id"
# tweet.user.name="Actual Name of User"
# tweet.user.screen_name="User displaed name in Twitter"
# tweet.text="Tweet of user containing specific keyword"
# tweet.user.location="Home Location of User"
# tweet.palce="In order to get tweet location"

## Creating a dataframe out of it
df1=pd.DataFrame(data,columns=["id","name","screen_name","text","home_location","tweet_location","mentioned_location"])

#Similar Kind of thing is done for key word like "Chennai",and "Mumbai"
new_search="Mumbai"+" -filter:retweets"
tweets=tp.Cursor(api.search,q=new_search,lang="en").items(700)
data=[[tweet.user.id,tweet.user.name,tweet.user.screen_name,tweet.text,tweet.user.location,tweet.place,"Mumbai"] for tweet in tweets]
df2=pd.DataFrame(data,columns=["id","name","screen_name","text","home_location","tweet_location","mentioned_location"])

new_search="Chennai"+" -filter:retweets"
tweets=tp.Cursor(api.search,q=new_search,lang="en").items(700)
data=[[tweet.user.id,tweet.user.name,tweet.user.screen_name,tweet.text,tweet.user.location,tweet.place,"Chennai"] for tweet in tweets]
df3=pd.DataFrame(data,columns=["id","name","screen_name","text","home_location","tweet_location","mentioned_location"])

## Let's Concatenate these 3 dataframes
df_whole=pd.concat([df1,df2,df3])

#Let's perform Basic preprocessing and Feature Engineering on extracted data

## Let's remove url from the end of each tweet
import re
def remove_url(x):
  return " ".join(re.sub("[^0-9A-Za-z \t]0 | (\w+:\/\/\S+)", "",x).split())

df_whole["text"]=df_whole["text"].apply(lambda x : remove_url(x))

## Let's extract tweet location
def extract_tweet_location(place):
    try:
      name=place.name
      return name
    except:
      return None
df_whole["tweet_location"]=df_whole["tweet_location"].apply(lambda x : extract_tweet_location(x))

## Let's Shuffle the dataset
df_whole=df_whole.sample(frac=1).reset_index(drop=True)
data=df_whole.copy()

## Let's Remove the rows who doesn't have home location
data2=data[data["home_location"].notnull()]

## Let's Fill the Nan value of tweet_location column by home_location
data2["tweet_location"]=np.where(data2["tweet_location"].isnull(),data2["home_location"],data2["tweet_location"])

## Let's remove the extra character from tweets like emojis ,punctuation
data2["text"]=data2["text"].apply(lambda x: re.sub("[^A-Za-z']",' ',x))

## Let's remove the extra character from home location and tweet location
data2["home_location"]=data2["home_location"].apply(lambda x: re.sub("[^A-Za-z]",' ',x).strip())
data2["tweet_location"]=data2["tweet_location"].apply(lambda x: re.sub("[^A-Za-z]",' ',x).strip())

## To remove Blank string By Nan Values
data2["home_location"].replace("",np.nan,inplace=True)

## Let's Remove The Nan values and Create a new dataFrame
data3=data2[data2["home_location"].notnull()]

## Let's Save it in a CSV file
data3.to_csv("Twitter_data.csv",index=0)