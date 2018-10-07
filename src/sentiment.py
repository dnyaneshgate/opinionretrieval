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
        self._df = None

    def load(self):
        print('Loading dataset...')
        self._df = pd.read_csv(self._filename, encoding=self._encoding, names=self._columns)

    def drop_columns(self, columns):
        print('Dropping cloumns: ', columns)
        self._df = self._df.drop(columns, axis=1)

    def drop_null_entries(self):
        print('Drop NULL entries...')
        self._df.dropna(inplace=True)
        self._df.reset_index(drop=True, inplace=True)

    def save(self, filepath, encoding='utf-8'):
        print('Saving dataset to file ', filepath)
        self._df.to_csv(filepath, encoding=encoding, index=False, header=False)

    def cleanup(self):
        print('Dataset preprocessing')
        self._df['text'] = self._df['text'].apply(self._preprocess_tweets)

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


dataset = TwitterDataset('dataset/preprocessed_twitter_dataset.csv', columns=['sentiment', 'text'])
dataset.load()
dataset.drop_null_entries()

# print('CountVectorizer:')
# cvec = CountVectorizer()
# cvec.fit(dataset._df['text'])
# print('CountVectorizer: extracted words: ', len(cvec.get_feature_names()))

# neg_doc_matrix = cvec.transform( dataset._df[ dataset._df.sentiment == 0 ].text )
# pos_doc_matrix = cvec.transform( dataset._df[ dataset._df.sentiment == 4 ].text )

# neg_tf = np.sum(neg_doc_matrix, axis=0)
# pos_tf = np.sum(pos_doc_matrix, axis=0)
# print('Neg TF: ', neg_tf)
# neg = np.squeeze(np.asarray(neg_tf))
# pos = np.squeeze(np.asarray(pos_tf))
# print('Neg: ', neg)
# term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
# print(term_freq_df.info())
# print(term_freq_df.head())



x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset._df['text'], dataset._df['sentiment'])

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

print('Extracting Features...')
tfidf_vec = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', encoding='utf-8', stop_words=stopwords.words('english'))
# tfidf_vec = TfidfVectorizer(encoding='utf-8', max_features=5000, stop_words=stopwords.words('english'))
tfidf_vec.fit(dataset._df['text'])

X_train = tfidf_vec.transform(x_train)
X_test = tfidf_vec.transform(x_test)

print('Training Multinomial Naive Bayes')
MNB_classifier = MultinomialNB()
MNB_classifier.fit( X_train, y_train )

print('Training Logistic Regression')
LR_classifier = LogisticRegression()
LR_classifier.fit( X_train, y_train )

predictions = MNB_classifier.predict(X_test)
accuracy = metrics.accuracy_score(predictions, y_test)
print('MNB Accuracy: ', accuracy)

predictions = LR_classifier.predict(X_test)
accuracy = metrics.accuracy_score(predictions, y_test)
print('LR Accuracy: ', accuracy)











