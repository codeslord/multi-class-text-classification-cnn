# coding: utf-8

"""
Embedding layer as an input to fully connected layers

@author: rohithrnair

"""
import multiprocessing
from io import StringIO
import numpy as np
import gzip
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import itertools
import gensim
from gensim.corpora.dictionary import Dictionary
import nltk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Flatten
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir=".")

np.random.seed(1992)
cpu_count = multiprocessing.cpu_count()
vocab_dim = 32
maxlen = 100
n_iterations = 10  # ideally more, since this improves the quality of the word vecs
n_exposures = 30
window_size = 7
batch_size = 10
n_epoch = 2
input_length = 1

def create_dictionaries(train = None,
                        target = None,
                        model = None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (train is not None) and (model is not None) and (target is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(data):
            ''' Words become integers
            '''
            for key in data.keys():
                txt = data[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data[key] = new_txt
            return data
        train = parse_dataset(train)
        tar = parse_dataset(target)
        return w2indx, w2vec, train, tar
    else:
        print('No data provided...')

def xpath_soup(element):
    """
    Generate xpath of soup element
    :param element: bs4 text or node
    :return: xpath as string
    """
    components = []
    child = element if element.name else element.parent
    for parent in child.parents:
        """
        @type parent: bs4.element.Tag
        """
        previous = itertools.islice(parent.children, 0, parent.contents.index(child))
        xpath_tag = child.name
        xpath_index = sum(1 for i in previous if i.name == xpath_tag) + 1
        components.append(xpath_tag if xpath_index == 1 else '%s[%d]' % (xpath_tag, xpath_index))
        child = parent
    components.reverse()
    return '/%s' % '/'.join(components)

page = urllib.request.urlopen("https://test.salesforce.com/")
gzipped = page.info().get('Content-Encoding') == 'gzip'
df = pd.DataFrame(columns=['element'])

if gzipped:
    buf  = StringIO(page.read())
    f = gzip.GzipFile(fileobj = buf)
    data = f.read()
else:
    data = page

soup = BeautifulSoup(data, 'lxml')
print(soup.original_encoding)
outstr_list = []
names_list = []
types_list = []
ids_list =[]
classes_list =[]
linktext_list = []
xpath_list = []
def walker(soup):
    if soup.name is not None:
        for child in soup.children:
            #process node
            try:
                outstr = str(child.name) + ":" + str(type(child)) + ":" + str(child.attrs)
                names = str(child.name)
                types = str(type(child))
                xpath = xpath_soup(child)
                if child.has_attr('id'):
                    ids = str(child['id'])
                else:
                    ids = None
                if child.has_attr('class'):
                    classes = str(child['class'])
                else:
                    classes = None                    
                if str(child.text):
                    linktext = str(child.text)   
                else:
                    linktext = None
            except AttributeError:
                outstr = str(child.name) + ":" + str(type(child))
                names = str(child.name)
                types = str(type(child))
                ids = None
                classes = None
                linktext = None
                xpath = None
            outstr_list.append(outstr)
            names_list.append(names)
            types_list.append(types)
            ids_list.append(ids)
            classes_list.append(classes)
            linktext_list.append(linktext)
            xpath_list.append(xpath)
            walker(child)
            
walker(soup)
table_data = []
table_data.append(names_list)
table_data.append(types_list)
table_data.append(ids_list)
table_data.append(classes_list)
table_data.append(linktext_list)
table_data.append(xpath_list)
# print(table_data)
df = pd.DataFrame(table_data)
df = df.transpose()
df.columns = ['element', 'types', 'id', 'classes', 'linktext', 'xpath']

data = df.copy()

df['class'] = df.classes.replace('\[', '', regex=True).replace('\]', '', regex=True).replace("'", '', regex=True).replace(',', ' ', regex=True)
df.drop(columns='classes', inplace=True)
df['type'] = df.types.replace("\<class 'bs4.element.", '', regex=True).replace("'\>", '', regex=True)
df.drop(columns='types', inplace=True)

df['full'] = df.type.fillna(value=' ') + ' ' + df.id.fillna(value=' ') + ' ' + df['class'].fillna(value=' ') + ' ' + df.linktext.fillna(value=' ') + ' ' + df.xpath.fillna(value=' ')

dataset = df.copy()

dataset = df[['element', 'full']]

dataset.to_csv('dataset.csv')