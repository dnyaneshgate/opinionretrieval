import logging
import logger
import twitter
import sys
import argparse

log = logging.getLogger(__name__)

def received_tweet(tweet):
    text, loc = tweet
    print(text)

def twitter_stream(params):
    stream = twitter.TwitterStream(params.auth_file, tweet_limit=50)
    stream.connect()
    stream.add_filter("TRUMP")
    stream.register_callback(callback=received_tweet)
    stream.start()

def twitter_search(params, search_text):
    server = twitter.Twitter(params.auth_file)
    server.connect()
    server.search(search_text, callback=received_tweet)

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('auth_file', help = 'Absolute path of the auth file')
    pargs = p.parse_args(argv)
    return pargs

def main():
    try:
        params = _parse_args(sys.argv[1:])
        # twitter_stream(params)
        twitter_search(params, 'samsung')
        input('Press Enter to exit..')
    except Exception as e:
        log.traceback(e)
        log.error('Exception: %s', str(e))

if __name__ == "__main__":
    log = logger.init_logger('sentiment.log')
    main()



