import logging
from datetime import datetime, timedelta
from abc import ABCMeta, abstractmethod
import jsonpickle
import json
from queue import Queue
import tweepy
from tweepy import Stream
from tweepy import StreamListener
import pprint as pp
import preprocessor

log = logging.getLogger()

class IPlatform(metaclass=ABCMeta):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def search(self, text: str, place: str, granularity: str, count: int) -> list:
        pass

class Granularity(object):
    COUNTRY = 'country'
    STATE = 'state'
    CITY = 'city'

class GeoLoc(object):
    def __init__(self):
        pass

class TimeFrame(object):
    DAY = 1
    WEEK = 7
    MONTH = 30
    YEAR = 365

    @staticmethod
    def get_query(timeframe):
        return (datetime.now() - timedelta(days=timeframe)).strftime("%Y-%m-%d")

class Message(object):
    def __init__(self, text: str, geoloc: GeoLoc):
        self.text = text
        self.geoloc = geoloc

class TweetListener(StreamListener):
    def __init__(self, queue, tweet_limit=-1, **kwargs):
        self.__queue = queue
        self.__tweet_limit = tweet_limit
        super(TweetListener, self).__init__(**kwargs)

    def on_data(self, raw_data):
        try:
            if self.__tweet_limit > 0:
                self.__tweet_limit -= 1
            if self.__tweet_limit == 0:
                log.info('Reached max tweet limit.')
                self.__queue.put(None)
                return False
            self.__queue.put(raw_data)
        except Exception as e:
            log.error('stream error : %s', str(e))
            log.traceback(e)
        return True

    def on_error(self, status_code):
        log.error("TweetListener::on_error : %s", status_code)
        return True

    def on_timeout(self):
        log.error("TweetListener::on_timeout : timeout...")
        return True

class Twitter(IPlatform):
    def __init__(self, auth_file):
        log.info('Twitter::__init__()')
        self._auth = None
        self._api = None
        with open(auth_file, 'r') as fp:
            self._auth_params = json.load(fp)

    def connect(self):
        log.info('Twitter::connect()')
        # self._auth = tweepy.AppAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        self._auth = tweepy.OAuthHandler(self._auth_params['TWITTER_API_KEY'], self._auth_params['TWITTER_API_SECRET'])
        self._auth.set_access_token(self._auth_params['TWITTER_ACCESS_TOKEN'], self._auth_params['TWITTER_ACCESS_TOKEN_SECRET'])
        self._api = tweepy.API(self._auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def search(self, text: str, timeframe: int = TimeFrame.DAY, near: str = 'India', max_tweets: int = 100, since_id: int = None, max_id: int = -1) -> list:
        log.info('Twitter::search()')
        result = []

        since = TimeFrame.get_query(timeframe)
        query = "\"{0}\" since:{1}".format(text, since)

        downloaded_tweets = 0
        tweet_count = 0

        kwargs = { "q": query, "lang": "en", "count": 100, "tweet_mode": "extended" }

        while tweet_count < max_tweets:
            try:
                if max_id >= 0:
                    kwargs.update( { "max_id": str(max_id - 1) } )
                if since_id:
                    kwargs.update( { "since_id": since_id } )

                tweets = self._api.search(**kwargs)

                if not tweets:
                    print("No new tweets.")
                    break

                for tweet in tweets:
                    # result.append( jsonpickle.encode(tweet._json, unpicklable=False) )
                    result.append( self._process_tweet(tweet._json) )

                tweet_count += len(tweets)
                max_id = tweets[-1].id
            except Exception as e:
                log.traceback(e)
                break
        return result

    def _process_tweet(self, tweet):
        text = self._get_text(tweet)
        loc = self._get_loc(tweet)
        return (text, loc)

    def _get_text(self, tweet):
        if "retweeted_status" in tweet:
            try:
                text = tweet['retweeted_status']['extended_tweet']['full_text']
            except:
                text = tweet['retweeted_status']['text']
        else:
            try:
                text = tweet['extended_tweet']['full_text']
            except:
                text = tweet['text']

        return self._filter_text(text)

    def _filter_text(self, text):
        return preprocessor.clean(text)

    def _get_loc(self, tweet):
        if tweet['coordinates']:
            coordinates = tweet['coordinates']
            loc = { 'geoloc': coordinates, 'type': 'coordinates' }
        elif tweet['place']:
            loc = { 'geoloc': tweet['place'], 'type': 'place' }
        else:
            loc = { 'geoloc': tweet['user']['location'], 'type': 'user_location' }

        return loc

class TwitterStream(Twitter):
    def __init__(self, auth_file, tweet_limit=-1):
        log.info('TwitterStream::__init__()')
        super(TwitterStream, self).__init__(auth_file)
        self.__tweet_limit = tweet_limit
        self.__queue = Queue()
        self.__tracks = []
        self.__langs = ['en']

    def add_filter(self, *track):
        log.info('TwitterStream::add_filter()')
        self.__tracks.extend(list(track))

    def _process_tweet(self, tweet):
        result = super(TwitterStream, self)._process_tweet(tweet)
        # log.info('retweet: %s', tweet['retweeted'])
        # if tweet['retweeted']:
        #     in_reply_to_status_id = tweet['in_reply_to_status_id']
        #     orig_tweet = self._api.get_status(in_reply_to_status_id)
        #     # orig_tweet = self._api.statuses_lookup(id_=[in_reply_to_status_id], tweet_mode='extended')
        #     log.info("orig_tweet: %s", orig_tweet._json)
        return result

    def start(self):
        log.info('TwitterStream::start()')
        listener = TweetListener(self.__queue, self.__tweet_limit, api=self._api)
        stream = Stream(auth=self._auth, listener=listener, tweet_mode='extended')
        stream.filter(track=self.__tracks, languages=self.__langs, async=True)
        while True:
            tweet = self.__queue.get()
            self.__queue.task_done()
            if not tweet:
                break
            tweet = json.loads(tweet)
            (text, loc) = self._process_tweet(tweet)
            log.info("-------------------------------------------------------------------")
            # log.info("TWEET: %s\n\n", pp.pformat(json.loads(tweet), indent=2, width=1))
            log.info("TEXT: %s", text)
            log.info("LOC: %s", loc)

__all__ = [ 'Twitter', 'TwitterStream' ]

