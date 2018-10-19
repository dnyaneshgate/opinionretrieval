import re
import pickle
import time
from bs4 import BeautifulSoup
from multiprocessing import Pool
from collections import namedtuple

import pandas as pd
import numpy as np
import preprocessor as tweetpp

from sklearn import model_selection, preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = tweetpp.clean(tweet)
    return tweet

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

                        lambda text: text.lower(),

                        # Remove negative contractions
                        lambda text: neg_pat.sub(lambda x: negative_contractions[x.group()], text),

                        # Letters only
                        lambda text: re.sub('[^a-zA-Z]', ' ', text)

                        # Remove white spaces
                        lambda text: ' '.join([x for x in tokenizer.tokenize(text) if len(x) > 1])
                   ]
    for func in cleanup_func:
        tweet = func(tweet)
    return tweet

class TwitterDataset(object):
    def __init__(self, filename, encoding='utf-8', columns=None):
        self._filename = filename
        self._encoding = encoding
        self._columns = columns
        self._word_tokenizer = WordPunctTokenizer()
        self.df = None

    def load(self):
        print('Loading dataset...')
        self.df = pd.read_csv(self._filename, encoding=self._encoding, names=self._columns)

    def drop_columns(self, columns):
        print('Dropping cloumns: ', columns)
        self.df = self.df.drop(columns, axis=1)

    def drop_null_entries(self):
        print('Drop NULL entries...')
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def save(self, filepath, encoding='utf-8'):
        print('Saving dataset to file ', filepath)
        self.df.to_csv(filepath, encoding=encoding, index=False, header=False)

    def cleanup(self):
        print('Dataset preprocessing')
        self.df['text'] = self.df['text'].apply(self._preprocess_tweets)

    def _preprocess_tweets(self, tweet):
        soup = BeautifulSoup(tweet, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode('utf-8-sig').replace(u'\ufffd', '?')
        except Exception:
            bom_removed = souped
        stripped = re.sub(mentions_pat, '', bom_removed)
        stripped = re.sub(url_pat, '', stripped)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pat.sub(lambda x: negative_contractions[x.group()], lower_case)
        letters_only = re.sub('[^a-zA-Z]', ' ', neg_handled)
        words = [ x for x in self._word_tokenizer.tokenize(letters_only) if len(x) > 1 ]
        return (' '.join(words)).strip()

def extract_features(cvec, nfeatures, x_train, x_test):
    print(cvec.__class__.__name__, ': ngram_range=', cvec.ngram_range, ',max_features:', nfeatures)
    stime = time.time()
    X_train = cvec.fit_transform(x_train)
    X_test  = cvec.transform(x_test)
    print(cvec.__class__.__name__, ': ngram_range=', cvec.ngram_range, ',max_features:', nfeatures, 'Time taken: ', time.time() - stime)
    return (nfeatures, X_train, X_test)

def train_classifier(classifier, nfeatures, X_train, y_train, X_test, y_test):
    stime = time.time()
    try:
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        return (nfeatures, metrics.accuracy_score(y_test, prediction))
    finally:
        print('Time taken by classifier ', classifier.__class__.__name__, ' : ', time.time() - stime)

def train(vectorizer):
    start = time.time()
    n_features = np.arange(10000, 100001, 10000)
    print('nfeatures: ', n_features)

    dataset = TwitterDataset('dataset/preprocessed_twitter_dataset.csv', columns=['sentiment', 'text'])
    dataset.load()
    dataset.drop_null_entries()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset.df['text'], dataset.df['sentiment'])

    stop_words = None

    cvec_ug = []
    cvec_bg = []
    cvec_tg = []

    pool = Pool(processes=8)

    ug_result = []
    bg_result = []
    tg_result = []
    for n in n_features:
        ug_cvec = vectorizer(encoding='UTF-8', max_features=n, stop_words=stop_words)
        ug_result.append( pool.apply_async(extract_features, (ug_cvec, n, x_train, x_test)) )

        bg_cvec = vectorizer(encoding='UTF-8', max_features=n, stop_words=stop_words, ngram_range=(1,2))
        bg_result.append( pool.apply_async(extract_features, (bg_cvec, n, x_train, x_test)) )

        tg_cvec = vectorizer(encoding='UTF-8', max_features=n, stop_words=stop_words, ngram_range=(1,3))
        tg_result.append( pool.apply_async(extract_features, (tg_cvec, n, x_train, x_test)) )


    for res in ug_result:
        cvec_ug.append( res.get() )
    for res in bg_result:
        cvec_bg.append( res.get() )
    for res in tg_result:
        cvec_tg.append( res.get() )

    classifiers = [
                        LogisticRegression(),
                        MultinomialNB(),
                        BernoulliNB(),
                        LinearSVC()
                  ]



    accuracy_map = {}
    for classifier in classifiers:
        print('Training classifier: ', classifier.__class__.__name__, ' , Unigram Feattures')
        ug_result = [ pool.apply_async(train_classifier, (classifier, n, X_train, y_train, X_test, y_test)) for n, X_train, X_test in cvec_ug ]

        print('Training classifier: ', classifier.__class__.__name__, ' , Bigram Feattures')
        bg_result = [ pool.apply_async(train_classifier, (classifier, n, X_train, y_train, X_test, y_test)) for n, X_train, X_test in cvec_bg ]

        print('Training classifier: ', classifier.__class__.__name__, ' , Trigram Feattures')
        tg_result = [ pool.apply_async(train_classifier, (classifier, n, X_train, y_train, X_test, y_test)) for n, X_train, X_test in cvec_tg ]

        accuracy_map['Unigram'] = [ res.get() for res in ug_result ]
        accuracy_map['Bigram'] = [ res.get() for res in bg_result ]
        accuracy_map['Trigram'] = [ res.get() for res in tg_result ]

        plt.figure(figsize=(8, 6))
        plt.title("%s - %s: Accuracy vs Feattures" % (classifier.__class__.__name__, vectorizer.__name__))
        plt.xlabel("Accuracy")
        plt.ylabel("Features")

        for ngram, result in accuracy_map.items():
            df = pd.DataFrame(result, columns=['nfeatures', 'accuracy'])
            plt.plot(df.nfeatures, df.accuracy, label="%s %s" % (vectorizer.__name__, ngram))

        plt.legend()
        plt.savefig("%s - %s - N-gram.png" % (classifier.__class__.__name__, vectorizer.__name__))

    print('Execution time: ', time.time() - start)



class Estimator(object):
    def __init__(self, clf, vect, pipeline = None):
        self._clf = clf
        self._vect = vect
        self._pipeline = pipeline

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
            return cls(None, None, pipeline)


    def fit_save(self, x_train, y_train, filepath):
        self._pipeline = Pipeline([ ('vect', self._vect), ('clf', self._clf) ])
        self._pipeline.fit(x_train, y_train)
        with open(filepath, 'wb') as f:
            pickle.dump(self._pipeline, f)

    def predict(self, x_test):
        return self._pipeline.predict(x_test)


def main1():
    # train(CountVectorizer)
    # train(TfidfVectorizer)

    stime = time.time()
    try:
        stop_words = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
 #stopwords.words('english')
        stemmer = SnowballStemmer('english')
        lemmer=WordNetLemmatizer()

        transformer_func = [
                                lambda text: ' '.join([word for word in text.split() if word not in stop_words]),
                                lambda text: ' '.join([stemmer.stem(word) for word in text.split()]),
                                lambda text: ' '.join([lemmer.lemmatize(word) for word in text.split()])
                           ]

        dataset = TwitterDataset('dataset/preprocessed_twitter_dataset.csv', columns=['sentiment', 'text'])
        dataset.load()
        dataset.drop_null_entries()

        # for func in transformer_func:
        #     dataset.df['text'] = dataset.df['text'].apply(func)
        #     print(dataset.df.text.head(5))

        x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset.df['text'], dataset.df['sentiment'])

        vect = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1,3))
        clf = LogisticRegression()

        X_train = vect.fit_transform(x_train)
        X_test = vect.transform(x_test)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

    finally:
        print('Time Taken By Application: {}'.format(time.time() - stime))

def main():



if __name__ == "__main__":
    # import logger
    # log = logger.init_logger()
    main()







