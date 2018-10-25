FROM continuumio/anaconda3:5.2.0

RUN pip install tweepy jsonpickle tweet-preprocessor

RUN [ "python", "-c", "import nltk; nltk.download('all')" ]

RUN /opt/conda/bin/conda install jupyter -y
RUN mkdir /opt/notebooks
# RUN /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root

RUN pip install pika
RUN pip install flask-restful

CMD python
