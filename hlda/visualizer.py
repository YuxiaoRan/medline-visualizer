# credit to joewandy/hlda
# modified by YuxiaoRan

import sys
basedir = '../'
sys.path.append(basedir)

import pylab as plt                             #require freetype, png
import nltk                                       #require sqlite3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords                 #require sqlite3
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from hlda.sampler import HierarchicalLDA
from IPython.core.display import HTML, display

from ipywidgets import interact
import ipywidgets as widgets

import _pickle as cPickle
import gzip

import string
import glob

nltk.download('stopwords')
nltk.download('punkt')


stopset = stopwords.words('english') + list(string.punctuation) + ['will', 'also', 'said']

corpus = []
all_docs = []
vocab = set()

stemmer = PorterStemmer()
for filename in glob.glob('../medline17n0010_s/*.txt'):
    with open(filename) as f:
        try:

            doc = f.read().splitlines()
            doc = filter(None, doc)  # remove empty string
            doc = '. '.join(doc)
            doc = doc.translate(string.punctuation)  # strip punctuations
            doc = doc.translate('0123456789')  # strip numbers
            doc = doc.encode('ascii', 'ignore')  # ignore fancy unicode chars
            all_docs.append(doc)

            tokens = word_tokenize(str(doc))
            filtered = []
            for w in tokens:
                w = stemmer.stem(w.lower())  # use Porter's stemmer
                if len(w) < 3:  # remove short tokens
                    continue
                if w in stopset:  # remove stop words
                    continue
                filtered.append(w)

            vocab.update(filtered)
            corpus.append(filtered)

        except UnicodeDecodeError:
            print ('Failed to load', filename)

vocab = sorted(list(vocab))
vocab_index = {}
for i, w in enumerate(vocab):
    vocab_index[w] = i


print (len(all_docs))


print (len(vocab))
print (vocab[0:100])

# Visualize the data
wordcloud = WordCloud(background_color='white').generate(b' '.join(all_docs).decode())
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Run hLDA
print (len(vocab), len(corpus), len(corpus[0]), len(corpus[1]))
    # Convert words into indices
new_corpus = []
for doc in corpus:
    new_doc = []
    for word in doc:
        word_idx = vocab_index[word]
        new_doc.append(word_idx)
    new_corpus.append(new_doc)

print (len(vocab), len(new_corpus))
print (corpus[0][0:10])
print (new_corpus[0][0:10])

    # Create hLDA object and run sampler
n_samples = 500       # no of iterations for the sampler
alpha = 10.0          # smoothing over level distributions
gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
eta = 0.1             # smoothing over topic-word distributions
num_levels = 4        # the number of levels in the tree
display_topics = 2   # the number of iterations between printing a brief summary of the topics so far
n_words = 10           # the number of most probable words to print for each topic after model estimation
with_weights = False  # whether to print the words with the weights

hlda = HierarchicalLDA(new_corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)


# Visualize results
colour_map = {
    0: 'blue',
    1: 'red',
    2: 'green'
}


def show_doc(d=0):
    node = hlda.document_leaves[d]
    path = []
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()

    n_words = 10
    with_weights = False
    for n in range(len(path)):
        node = path[n]
        colour = colour_map[n]
        msg = 'Level %d Topic %d: ' % (node.level, node.node_id)
        msg += node.get_top_words(n_words, with_weights)
        output = '<h%d><span style="color:%s">%s</span></h3>' % (n + 1, colour, msg)
        display(HTML(output))

    display(HTML('<hr/><h5>Processed Document</h5>'))

    doc = corpus[d]
    output = ''
    for n in range(len(doc)):
        w = doc[n]
        l = hlda.levels[d][n]
        colour = colour_map[l]
        output += '<span style="color:%s">%s</span> ' % (colour, w)
    display(HTML(output))

interact(show_doc, d=(0, len(corpus)-1))

############
