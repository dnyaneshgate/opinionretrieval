FROM continuumio/anaconda3:5.2.0

RUN pip install tweepy jsonpickle tweet-preprocessor

RUN [ "python", "-c", "import nltk; nltk.download('all')" ]

CMD python
