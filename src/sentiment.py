import os
import re
import pickle
import time
import logging
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


log = logging.getLogger()

negative_contractions = {
    "can't": "can not",
    "don't": "do not",
    "isn't": "is not",
    "won't": "will not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "aren't": "are not",
    "doesn't": "does not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "mustn't": "must not",
    "didn't": "did not",
    "mightn't": "might not",
    "needn't": "need not",
}

mentions_pat = r'@[A-Za-z0-9_]+'
url_pat = r'https?://[^ ]+'
www_pat = r'www.[^ ]+'
neg_pat = re.compile(r'\b(' + '|'.join(negative_contractions.keys()) + r')\b')

tokenizer = WordPunctTokenizer()
stemmer = SnowballStemmer('english')
lemmer=WordNetLemmatizer()

def decode_utf(text):
    try:
        return text.decode('utf-8-sig').replace(u'\ufffd', '?')
    except:
        return text

def tweet_cleanup(tweet):
    cleanup_func = [
                        # HTML decoding
                        lambda text: BeautifulSoup(text, 'lxml').get_text(),

                        # Unicode decoding
                        lambda text: decode_utf(text),

                        # Removed mentions
                        lambda text: re.sub(mentions_pat, '', text),

                        # Removed URL
                        lambda text: re.sub(url_pat, '', text),

                        # Removed URL
                        lambda text: re.sub(www_pat, '', text),

                        # Convert to lower case
                        lambda text: text.lower(),

                        # Remove negative contractions
                        lambda text: neg_pat.sub(lambda x: negative_contractions[x.group()], text),

                        # Letters only
                        lambda text: re.sub('[^a-zA-Z]', ' ', text),

                        # Remove white spaces
                        lambda text: ' '.join([x for x in tokenizer.tokenize(text) if len(x) > 1]),

                        # # remove stop words
                        # lambda text: ' '.join([word for word in text.split() if word not in stop_words]),

                        # # apply stemmer
                        # lambda text: ' '.join([stemmer.stem(word) for word in text.split()]),

                        # #apply lemmatization
                        # lambda text: ' '.join([lemmer.lemmatize(word) for word in text.split()])
                   ]
    for func in cleanup_func:
        tweet = func(tweet)
    return tweet

def load_class(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def dump_class(class_, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(class_, f)

class TwitterDataset(object):
    def __init__(self, filename, encoding='utf-8', columns=None):
        self._filename = filename
        self._encoding = encoding
        self._columns = columns
        self._word_tokenizer = WordPunctTokenizer()
        self.df = None

    def load(self):
        log.info('Loading dataset...')
        self.df = pd.read_csv(self._filename, encoding=self._encoding, names=self._columns)

    def drop_columns(self, columns):
        log.info('Dropping cloumns: %s', columns)
        self.df = self.df.drop(columns, axis=1)

    def drop_null_entries(self):
        log.info('Drop NULL entries...')
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def save(self, filepath, encoding='utf-8'):
        log.info('Saving dataset to file %s', filepath)
        self.df.to_csv(filepath, encoding=encoding, index=False, header=False)

    def cleanup(self):
        log.info('Dataset preprocessing')
        self.df['text'] = self.df['text'].apply(tweet_cleanup)

class Estimator(object):
    def __init__(self, model_name, classifier=None, feature_extractor=None, pipeline=None):
        self._classifier = classifier
        self._feature_extractor = feature_extractor
        self._pipeline = pipeline
        self._model_name = model_name

    @classmethod
    def load_from_file(cls, clf_name, filepath):
        log.info('Loading pipeline from file %s', filepath)
        with open(filepath, 'rb') as f:
            return cls(clf_name, pipeline=pickle.load(f))

    def save(self, filepath):
        log.info('Saving pipeline to file %s', filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self._pipeline, f)

    @property
    def pipeline(self):
        return self._pipeline

    def fit(self, x_train, y_train):
        log.info('[%s] Training model...', self._model_name)
        self._pipeline = Pipeline(
            [
                ('transformer', self._feature_extractor),
                ('classifier', self._classifier)
            ]
        )
        self._pipeline.fit(x_train, y_train)

    def predict(self, x_test):
        return self._pipeline.predict(x_test)

class EstimatorPipeline(object):
    def ___init__(self):
        self._estimators = []

    def add(self, step, estimator):
        self._estimators.append((step, estimator))

class Sentiment(object):
    SENTIMENT_POSITIVE = 4
    SENTIMENT_NEGATIVE = 0
    SENTIMENT_MAP = {
        SENTIMENT_POSITIVE: 'Positive :-)',
        SENTIMENT_NEGATIVE: 'Negative :-('
    }
    def __init__(self):
        self._models = {
            'LogisticRegression': {
                'classifier': LogisticRegression(),
                'transformer': TfidfVectorizer(max_features=100000, ngram_range=(1,3), stop_words=None),
                'file': 'logistic_regression.pickle'
            }
        }
        self._estimators = []
        self._classifier = None

    def _train(self, force=False):
        log.info('Sentiment: _train()')
        training_dataset = TwitterDataset('dataset/training.1600000.processed.noemoticon.csv', columns=['sentiment', 'text'])
        training_dataset.load()
        testing_dataset = TwitterDataset('dataset/testdata.manual.2009.06.14.csv', columns=['sentiment', 'text'])
        testing_dataset.load()

        log.info(training_dataset.df)

        x_train, y_train = training_dataset.df.text.values.astype('U'), training_dataset.df.sentiment
        x_test, y_test = testing_dataset.df.text.values.astype('U'), testing_dataset.df.sentiment

        feature_union = FeatureUnion(
            [
                ('tfidf', TfidfVectorizer(max_features=100000, ngram_range=(1,3), stop_words=None)),
            ]
        )

        estimators = [
            ('pipe1', LogisticRegression()),
        ]

        pipeline = Pipeline(
            [
                ('features', feature_union),
                ('classifier', VotingClassifier(estimators=estimators))
            ]
        )
        pipeline.fit(x_train, y_train)
        log.info('score: %s', pipeline.score(x_test, y_test))

        self._classifier = pipeline

        dump_class(pipeline, 'voting_classifier.pickle')

    def prepare(self):
        log.info('Sentiment: prepare()')
        # for clf_name, params in self._models.items():
        #     if not os.path.exists(params['file']):
        #         self._train()
        #     estimator = Estimator.load_from_file(clf_name, params['file'])
        #     self._estimators.append((clf_name, estimator))
        filepath = 'voting_classifier.pickle'
        if os.path.exists(filepath):
            self._classifier = load_class(filepath)
        else:
            self._train()


    def get_sentiment(self, text):
        log.info('Sentiment: get_sentiment()')
        if not isinstance(text, list):
            text = [text]
        x_pred = pd.DataFrame(text, columns=['text'])
        x_pred['text'] = x_pred['text'].apply(tweet_cleanup)
        sentiment = self._classifier.predict(x_pred['text'])
        return sentiment

if __name__ == '__main__':
    console = logging.StreamHandler()
    log.addHandler(console)
    log.setLevel(logging.DEBUG)

    # cols = ['sentiment','id','date','query_string','user','text']
    # dataset = TwitterDataset('dataset/trainingandtestdata/testdata.manual.2009.06.14.csv', columns=cols, encoding='latin1')
    # dataset.load()
    # dataset.drop_columns(['id','date','query_string','user'])
    # dataset.cleanup()
    # dataset.drop_null_entries()
    # dataset.df = dataset.df[dataset.df.sentiment != 2]
    # log.info(dataset.df)
    # dataset.save('dataset/testdata.manual.2009.06.14.csv')

    analyzer = Sentiment()
    analyzer.prepare()
    data = [
        'This is a great movie',
        'worst movie evet watched'
    ]
    log.info('sentiment: %s', analyzer.get_sentiment(data))
    # log.info('sentiment: %s', analyzer.get_sentiment('this is very good movie'))
    # log.info('sentiment: %s', analyzer.get_sentiment('this is very bad movie'))