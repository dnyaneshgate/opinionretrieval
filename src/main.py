import logging
import logger
import twitter
import sys
import argparse

log = logging.getLogger(__name__)

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('auth_file', help = 'Absolute path of the auth file')
    pargs = p.parse_args(argv)
    return pargs

def main():
    try:
        params = _parse_args(sys.argv[1:])
        stream = twitter.TwitterStream(params.auth_file, tweet_limit=50)
        stream.connect()
        stream.add_filter("TRUMP")
        stream.start()
    except Exception as e:
        log.traceback(e)
        log.error('Exception: %s', str(e))

if __name__ == "__main__":
    log = logger.init_logger('sentiment.log')
    main()



