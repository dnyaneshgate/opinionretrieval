import os
import re
import pickle
import time
import logging
from bs4 import BeautifulSoup

import pandas as pd
# import numpy as np

from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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

wordnet_pos = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'N': wordnet.NOUN,
    'R': wordnet.ADV,
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

def lemmatize(text):
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    lemm_words = [ lemmer.lemmatize( w[0], wordnet_pos.get(w[1], wordnet.NOUN)) for w in word_pos ]
    return ' '.join(lemm_words)

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

                        # apply stemmer
                        lambda text: ' '.join([stemmer.stem(word) for word in text.split()]),

                        # apply lemmatization
                        lambda text: lemmatize(text),

                        lambda text: str(text),
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

    def train(self, training_dataset, testing_dataset, classifier_path):
        t0 = time.time()
        log.info('Sentiment: train()')

        x_train, y_train = training_dataset.df.text.values.astype('U'), training_dataset.df.sentiment
        x_test, y_test = testing_dataset.df.text.values.astype('U'), testing_dataset.df.sentiment

        feature_union = FeatureUnion(
            [
                ('tfidf', TfidfVectorizer(max_features=100000, ngram_range=(1,3), stop_words=None)),
            ]
        )

        estimators = [
            ('pipe1', LogisticRegression()),
            ('pipe2', MultinomialNB()),
            ('pipe3', LinearSVC()),
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

        dump_class(pipeline, classifier_path)

        log.info('Sentiment: train() : time spent: %s', time.time() - t0)

    def load(self, classifier_path):
        log.info('Sentiment: load()')
        self._classifier = load_class(classifier_path)

    def get_sentiment(self, text):
        log.info('Sentiment: get_sentiment()')
        if not isinstance(text, list):
            text = [text]
        x_pred = pd.DataFrame(text, columns=['text'])
        x_pred['text'] = x_pred['text'].apply(tweet_cleanup)
        sentiment = self._classifier.predict(x_pred['text'])
        return sentiment


def clean_and_save_dataset(filepath):
    t0 = time.time()
    log.info('Cleanup : %s', filepath)
    cols = ['sentiment','id','date','query_string','user','text']
    dataset = TwitterDataset(filepath, columns=cols, encoding='latin1')
    dataset.load()
    dataset.drop_columns(['id','date','query_string','user'])
    dataset.cleanup()
    dataset.drop_null_entries()
    dataset.df = dataset.df[dataset.df.sentiment != 2]
    log.info(dataset.df)
    destfile = os.path.basename(filepath)
    destpath = os.path.dirname(filepath)
    destfilepath = os.path.join(destpath, 'preprocessed_' + destfile)
    log.info('Save : %s', destfilepath)
    dataset.save(destfilepath)
    log.info('clean_and_save_dataset: time spent: %s', time.time() - t0)

def train_classifier(classifier_path):
    log.info('Training classifier')
    # dataset_dir = 'dataset'
    # training_dataset_file = 'training.1600000.processed.noemoticon.csv'
    # testing_dataset_file = 'testdata.manual.2009.06.14.csv'

    # log.info('Cleanup Training Dataset...')
    # clean_and_save_dataset(os.path.join(dataset_dir, training_dataset_file))
    # log.info('Cleanup Testing Dataset...')
    # clean_and_save_dataset(os.path.join(dataset_dir, testing_dataset_file))

    dataset_dir = 'dataset'
    training_dataset_file = 'preprocessed_training.1600000.processed.noemoticon.csv'
    testing_dataset_file = 'preprocessed_testdata.manual.2009.06.14.csv'

    training_dataset = TwitterDataset(os.path.join(dataset_dir, training_dataset_file), columns=['sentiment', 'text'])
    training_dataset.load()

    testing_dataset = TwitterDataset(os.path.join(dataset_dir, testing_dataset_file), columns=['sentiment', 'text'])
    testing_dataset.load()

    sentiment = Sentiment()
    sentiment.train(training_dataset, testing_dataset, classifier_path)

def _init_logging():
    console = logging.StreamHandler()
    log.addHandler(console)
    log.setLevel(logging.DEBUG)

if __name__ == '__main__':
    _init_logging()

    classifier_path = 'sentiment_classifier.pickle'
    # train_classifier(classifier_path)

    data = [
        ('So there is no way for me to plug it in here in the US unless I go by a converter.', 0),
        ('Good case, Excellent value.',4),
        ('Great for the jawbone.',4),
        ('Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!!',0),
        ('The mic is great.',4),
        ('I have to jiggle the plug to get it to line up right to get decent volume.',0),
        ('If you have several dozen or several hundred contacts, then imagine the fun of sending each of them one by one.',0),
        ('If you are Razr owner...you must have this!',4),
        ('Needless to say, I wasted my money.',0),
        ('What a waste of money and time!.',0),
    ]


    sentiment = Sentiment()
    sentiment.load(classifier_path)
    prediction = sentiment.get_sentiment( [ text for (text, _) in data ] )

    log.info('sentiment: %s', prediction)

