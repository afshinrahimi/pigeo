'''
Created on 1 Apr 2016

@author: af
'''
import tweepy
from tweepy import OAuthHandler
import logging
import pdb

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

consumer_key = 'COVV87zJN2wHRfAz7zB5p2QPQ'
consumer_secret = 'HhNmLOsIui0rG04XltDSHdmEBNf9IAtkZeW17U7pFYAuKf8qiv'
access_token = '2205031009-a7FMWRzzTi5wooSMFkYUqyiq1aGREBSMCyBX2vw'
access_token_secret = 'TfPQ5V8X9BOwjWQU7UBTJHTR8kYJzyhM1em8I4YGIcZxh'
# To get your own keys go to http://dev.twitter.com and create an app.
# The consumer key and secret will be generated for you after        
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

  
def download_user_tweets(user_screen_name, count=100):
    logging.debug('downloading Twitter timeline of user ' + user_screen_name)
    timeline = []
    try:
        timeline = api.user_timeline(user_screen_name, count=count)
    except:
        logging.error('Note that the consumer_key, consumer_secret, access_token and access_secret should be set in tweet_downloader.py source file.')    
    return timeline

def download_user_tweets_iterable(user_screen_names, count=100):
    
    timelines = {}
    for user in user_screen_names:
        try:
            timeline = api.user_timeline(user, count=count)
            timelines[user] = timelines
        except:
            logging.error('Note that the consumer_key, consumer_secret, access_token and access_secret should be set in tweet_downloader.py source file.')
            

    return timelines

if __name__ == '__main__':
    timeline = download_user_tweets('@afshinray')