import re
import pickle
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import preprocessor as tweetpp

from sklearn import model_selection, preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model  import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer


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





# dataset = TwitterDataset('dataset/trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='latin1', columns=['sentiment','id','date','flag','user','text'])
# dataset.load()
# dataset.drop_columns(['id', 'date', 'flag', 'user'])
# dataset.cleanup()
# dataset.save('dataset/trainingandtestdata/preprocessed_twitter_dataset.csv')


# dataset = TwitterDataset('dataset/preprocessed_twitter_dataset.csv', columns=['sentiment', 'text'])
# dataset.load()
# dataset.drop_null_entries()


# x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset.df['text'], dataset.df['sentiment'])

# encoder = preprocessing.LabelEncoder()
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.fit_transform(y_test)

# print('Extracting Features...')
# tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', encoding='utf-8', stop_words=stopwords.words('english'))
# # tfidf_vec = TfidfVectorizer(encoding='utf-8', max_features=5000, stop_words=stopwords.words('english'))
# tfidf_vec.fit(dataset.df['text'])

# X_train = tfidf_vec.transform(x_train)
# X_test = tfidf_vec.transform(x_test)

# print('Training Multinomial Naive Bayes')
# MNB_classifier = MultinomialNB()
# MNB_classifier.fit( X_train, y_train )

# print('Training Logistic Regression')
# LR_classifier = LogisticRegression()
# LR_classifier.fit( X_train, y_train )

# predictions = MNB_classifier.predict(X_test)
# accuracy = metrics.accuracy_score(predictions, y_test)
# print('MNB Accuracy: ', accuracy)

# predictions = LR_classifier.predict(X_test)
# accuracy = metrics.accuracy_score(predictions, y_test)
# print('LR Accuracy: ', accuracy)



def prepare_pipeline(classifier, vectorizer, n_features, ngram_range=(1,1), stop_words=None, **kwargs):
    pipelines = []
    for n in n_features:
        vec = vectorizer()
        vec.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range, **kwargs)
        pipeline = Pipeline( [
                    ('vectorizer', vec),
                    ('classifier', classifier)
                ] )
        pipelines.append((n, pipeline))
    return pipelines

def accuracy_summary(pipline, x_train, y_train, x_test, y_test):
    pipline.fit(x_train, y_train)
    prediction = pipline.predict(x_test)
    return metrics.accuracy_score(prediction, y_test)

def plot(title, axis_label, data):
    plt.figure(figsize=(8,6))
    for row in data:
        plt.plot(row[0], row[1], label=row[2])
    plt.title(title)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.legend()
    plt.savefig("N-gram_accuracy.png")

def main():
    n_features = np.arange(10000, 100001, 10000)
    lr_cvec_ug_pipeline = prepare_pipeline(LogisticRegression(), CountVectorizer, n_features)
    lr_cvec_bg_pipeline = prepare_pipeline(LogisticRegression(), CountVectorizer, n_features, ngram_range=(1,2))
    lr_cvec_tg_pipeline = prepare_pipeline(LogisticRegression(), CountVectorizer, n_features, ngram_range=(1,3))


    dataset = TwitterDataset('dataset/preprocessed_twitter_dataset.csv', columns=['sentiment', 'text'])
    dataset.load()
    dataset.drop_null_entries()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset.df['text'], dataset.df['sentiment'])

    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    lr_cvec_ug = [ (n_feature, accuracy_summary(pipeline, x_train, y_train, x_test, y_test)) for (n_feature, pipeline) in lr_cvec_ug_pipeline ]
    lr_cvec_bg = [ (n_feature, accuracy_summary(pipeline, x_train, y_train, x_test, y_test)) for (n_feature, pipeline) in lr_cvec_bg_pipeline ]
    lr_cvec_tg = [ (n_feature, accuracy_summary(pipeline, x_train, y_train, x_test, y_test)) for (n_feature, pipeline) in lr_cvec_tg_pipeline ]

    lr_cvec_ug_plot = pd.DataFrame(lr_cvec_ug, columns=['nfeatures', 'accuracy'])
    lr_cvec_bg_plot = pd.DataFrame(lr_cvec_bg, columns=['nfeatures', 'accuracy'])
    lr_cvec_tg_plot = pd.DataFrame(lr_cvec_tg, columns=['nfeatures', 'accuracy'])

    graph = []
    graph.append((lr_cvec_ug_plot.nfeatures, lr_cvec_ug_plot.accuracy, 'unigram count vectorizer'))
    graph.append((lr_cvec_bg_plot.nfeatures, lr_cvec_bg_plot.accuracy, 'bigram count vectorizer'))
    graph.append((lr_cvec_tg_plot.nfeatures, lr_cvec_tg_plot.accuracy, 'trigram count vectorizer'))

    plot( "Test Result: Accuracy", ("N Features", "Accuracy"), graph )


if __name__ == "__main__":
    # import logger
    # log = logger.init_logger()
    main()







