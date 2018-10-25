import pika

class MessageQueue(object):
    def __init__(self, host='rabbitmq'):
        self._host = host
        self._queues = {}
        self._callback = lambda: None

    def init(self):
        self._connection = pika.BlockingConnection( pika.ConnectionParameters(host=self._host) )
        self._channel = self._connection.channel()

    def declare_queue(self, queue):
        self._channel.queue_declare(queue=queue)

    def put(self, queue, text):
        self._channel.basic_publish(exchange='', routing_key=queue, body=text)

    def _cb_consume(self, channel, method, properties, body):
        self._callback(body)

    def start_consuming(self, callback, queue):
        self._callback = callback
        self._channel.basic_consume(self._cb_consume, queue=queue, no_ack=True)
        self._channel.start_consuming()

    def close(self):
        self._connection.close()


if __name__ == "__main__":
    import sys
    user = sys.argv[1]

    queue = MessageQueue()
    queue.init()
    queue.declare_queue('tweet')

    def received_tweet(tweet):
        print('received tweet: ', tweet)

    try:
        if user == 'producer':
            tweet = sys.argv[2]
            queue.put(queue='tweet', text=tweet)
        elif user == 'consumer':
            queue.start_consuming(callback=received_tweet, queue='tweet')
    finally:
        queue.close()

