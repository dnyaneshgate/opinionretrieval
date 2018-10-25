import sys

srcfile = sys.argv[1]
destfile = sys.argv[2]

srcfd = open(srcfile, 'r')
destfd = open(destfile, 'a')

while True:
    line = srcfd.readline()
    if not line:
        break
    line.replace('\n', '')
    text, sentiment = line.split('\t')
    destfd.write(','.join([text.strip(), sentiment.strip()]))
    destfd.write('\n')


srcfd.close()
destfd.close()
