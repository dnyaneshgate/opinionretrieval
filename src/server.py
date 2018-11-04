import os
import logging
from flask import Flask, request
from flask import render_template
from flask_restful import Resource, Api
import pprint as pp

from search import Search
from sentiment import Sentiment
import logger

log = logger.init_logger('sentiment.log')

template_folder = os.path.dirname(__file__)
template_folder = os.path.join(template_folder, 'templates')

app = Flask('OpinionRetrieval', template_folder=template_folder)

search_engine = Search('TwitterAuthToken.json')

classifier_path = 'sentiment_classifier.pickle'
sentiment = Sentiment()
sentiment.load(classifier_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods = ['POST'])
def search():
    search_text = request.form['search_text']
    tweets = search_engine.query(search_text)

    prediction = []
    sentiment_prediction = sentiment.get_sentiment( [ text for (_, text) in tweets ] )
    for (user, text), opinion in zip(tweets, sentiment_prediction):
        prediction.append( (user, text, opinion) )

    total = len(prediction)
    pos = len( [ x for x in prediction if x[2] == 4 ] )
    neg = len( [ x for x in prediction if x[2] == 0 ] )

    pos_percent = (pos * 100.0) / total
    neg_percent = (neg * 100.0) / total

    return render_template('sentiment.html', tweets=tweets, positive_percent=pos_percent, negative_percent=neg_percent)

def main():
    app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    main()
    # tweets = search_engine.query('Unity')

    # prediction = []
    # sentiment_prediction = sentiment.get_sentiment( [ text for (_, text) in tweets ] )
    # for (user, text), opinion in zip(tweets, sentiment_prediction):
    #     prediction.append( (user, text, opinion) )

    # total = len(prediction)
    # pos = len( [ x for x in prediction if x[2] == 4 ] )
    # neg = len( [ x for x in prediction if x[2] == 0 ] )

    # pos_percent = (pos * 100.0) / total
    # neg_percent = (neg * 100.0) / total

    # print('pos_percent: %s' % (pos_percent))
    # print('neg_percent: %s' % (neg_percent))
    # pp.pprint(prediction)