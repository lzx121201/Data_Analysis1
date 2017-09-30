from __future__ import division
from collections import Counter
import math, random, csv, json
import os
from bs4 import BeautifulSoup
import requests


from twython import Twython

# fill these in if you want to use the code
CONSUMER_KEY = "NCnIo0MDu3T2aw2aJoXLaj4wg"
CONSUMER_SECRET = "vMXJa1WTS8B0veqBFkVijzezEBLxKYwZ5Yv3gdYxiM7jM9kFow"
ACCESS_TOKEN = "800733264486895616-N27x96VmL0fJswneXNFcoWPlerGyc4o"
ACCESS_TOKEN_SECRET = "ihvgZa8D4senE65HGmWowidhanGIIhksigkrYjwYuCMRd"

def call_twitter_search_api():

    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)

    # search for tweets containing the phrase "amazing"
    for status in twitter.search(q='"amazing"')["statuses"]:
        user = status["user"]["screen_name"].encode('utf-8')
        text = status["text"].encode('utf-8')
        print(user, ":", text)
        print()

from twython import TwythonStreamer

# appending data to a global variable is pretty poor form
# but it makes the example much simpler
tweets = []

class MyStreamer(TwythonStreamer):
    """our own subclass of TwythonStreamer that specifies
    how to interact with the stream"""

    def on_success(self, data):
        """what do we do when twitter sends us data?
        here data will be a Python object representing a tweet"""

        # only want to collect English-language tweets
        if data['lang'] == 'en':
            tweets.append(data)
            print("received tweet #", len(tweets))
            saveFile = open(os.path.expanduser("~/Desktop/DataA/DA_final_project/tweets1.txt"),'a', encoding='utf-8')
            saveFile.write(str(data))
            saveFile.write('\n')
            saveFile.close()

        # stop when we've collected enough
        if len(tweets) >= 2000:
            self.disconnect()

    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()

def call_twitter_streaming_api():
    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
                        ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # starts consuming public statuses that contain the keyword 'Coldplay'
    stream.statuses.filter(track='Coldplay')

call_twitter_search_api()
call_twitter_streaming_api()

print(tweets)
