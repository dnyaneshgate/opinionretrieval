import logging
import threading
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
        self._auth = tweepy.AppAuthHandler(self._auth_params['TWITTER_API_KEY'], self._auth_params['TWITTER_API_SECRET'])
        # self._auth = tweepy.OAuthHandler(self._auth_params['TWITTER_API_KEY'], self._auth_params['TWITTER_API_SECRET'])
        # self._auth.set_access_token(self._auth_params['TWITTER_ACCESS_TOKEN'], self._auth_params['TWITTER_ACCESS_TOKEN_SECRET'])
        self._api = tweepy.API(self._auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def search(self, text: str, timeframe: int = TimeFrame.DAY, near: str = 'India', max_tweets: int = 100, since_id: int = None, max_id: int = -1, callback = None) -> list:
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
                    try:
                        processed_tweet = self._process_tweet(tweet._json)
                        if callback:
                            callback(processed_tweet)
                        else:
                            result.append( processed_tweet )
                    except Exception as e:
                        log.error('search(): tweet: %s', tweet)
                        log.error('search(): exception: %s', str(e))
                        log.traceback(e)

                tweet_count += len(tweets)
                max_id = tweets[-1].id
            except Exception as e:
                log.traceback(e)
                break
        return result

    def _process_tweet(self, tweet):
        # pp.pprint(tweet)
        text = self._get_text(tweet)
        user = self._get_user(tweet)
        # loc = self._get_loc(tweet)
        # return (text, loc)
        return (user, text)

    def _get_text(self, tweet_dict):
        try:
            tweet = tweet_dict
            if "retweeted_status" in tweet_dict:
                tweet = tweet_dict['retweeted_status']

            if 'extended_tweet' in tweet:
                tweet = tweet['extended_tweet']

            try:
                text = tweet['full_text']
            except:
                text = tweet['text']
        except Exception as e:
            log.debug('Tweet: %s', pp.pformat(tweet, indent=2, width=1))
            log.error('_get_text() exception: %s', str(e))
            log.traceback(e)
            raise
        return text

    def _get_user(self, tweet):
        user = tweet['user']
        ret = user['name'] + ' ' + user['created_at']
        return ret

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
        self.__thread = None

    def connect(self):
        log.info('Twitter::connect()')
        self._auth = tweepy.OAuthHandler(self._auth_params['TWITTER_API_KEY'], self._auth_params['TWITTER_API_SECRET'])
        self._auth.set_access_token(self._auth_params['TWITTER_ACCESS_TOKEN'], self._auth_params['TWITTER_ACCESS_TOKEN_SECRET'])
        self._api = tweepy.API(self._auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def add_filter(self, *track):
        log.info('TwitterStream::add_filter()')
        self.__tracks.extend(list(track))

    def register_callback(self, callback):
        self._callback = callback

    def _process_tweet(self, tweet):
        result = super(TwitterStream, self)._process_tweet(tweet)
        # log.info('retweet: %s', tweet['retweeted'])
        # if tweet['retweeted']:
        #     in_reply_to_status_id = tweet['in_reply_to_status_id']
        #     orig_tweet = self._api.get_status(in_reply_to_status_id)
        #     # orig_tweet = self._api.statuses_lookup(id_=[in_reply_to_status_id], tweet_mode='extended')
        #     log.info("orig_tweet: %s", orig_tweet._json)
        return result

    def _consume_tweets(self):
        while True:
            tweet = self.__queue.get()
            self.__queue.task_done()
            if not tweet:
                break
            tweet = json.loads(tweet)
            tweet = self._process_tweet(tweet)
            self._callback(tweet)

    def start(self):
        log.info('TwitterStream::start()')
        listener = TweetListener(self.__queue, self.__tweet_limit, api=self._api)
        stream = Stream(auth=self._auth, listener=listener, tweet_mode='extended')
        stream.filter(track=self.__tracks, languages=self.__langs, async=True)
        self.__thread = threading.Thread(target=self._consume_tweets)
        self.__thread.start()
        self.__thread.join()

__all__ = [ 'Twitter', 'TwitterStream' ]

