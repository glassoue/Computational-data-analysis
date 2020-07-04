#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:20:12 2020

@author: ghassen97
"""

import numpy as np
import pandas as pd
import re, nltk, gensim , spacy,string

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stop_words = stopwords.words('danish')

import stopwordsiso as stopwords
stopwords.langs()  # return a set of all the supported languages
stopwords.has_lang("da")  # check if there is a stopwords for the language
stopwords.stopwords("da")  # danish stopwords


import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
#from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
#%%
'''change dataset to get the corresponding results '''
data1 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20091.csv', error_bad_lines=False);
# data2 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20101.csv', error_bad_lines=False);
# data3 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20102.csv', error_bad_lines=False);
# data4 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20111.csv', error_bad_lines=False);
# data5 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20121.csv', error_bad_lines=False);
# data6 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20131.csv', error_bad_lines=False);
# data7 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20141.csv', error_bad_lines=False);
# data8 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20142.csv', error_bad_lines=False);
# data9 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20151.csv', error_bad_lines=False);
# data10 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20161.csv', error_bad_lines=False);

data1 = data1[['Speeces']].astype('str')
# data2 = data2[['Speeces']].astype('str')
# data3 = data3[['Speeces']].astype('str')
# data4 = data4[['Speeces']].astype('str')
# data5 = data5[['Speeces']].astype('str')
# data6 = data6[['Speeces']].astype('str')
# data7 = data7[['Speeces']].astype('str')
# data8 = data8[['Speeces']].astype('str')
# data9 = data9[['Speeces']].astype('str')
# data10 = data10[['Speeces']].astype('str')

#data_text = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data1 = pd.DataFrame(data1.Speeces.apply(lambda x: clean_text(x)))
# data2 = pd.DataFrame(data2.Speeces.apply(lambda x: clean_text(x)))
# data3 = pd.DataFrame(data3.Speeces.apply(lambda x: clean_text(x)))
# data4 = pd.DataFrame(data4.Speeces.apply(lambda x: clean_text(x)))
# data5 = pd.DataFrame(data5.Speeces.apply(lambda x: clean_text(x)))
# data6 = pd.DataFrame(data6.Speeces.apply(lambda x: clean_text(x)))
# data7 = pd.DataFrame(data7.Speeces.apply(lambda x: clean_text(x)))
# data8 = pd.DataFrame(data8.Speeces.apply(lambda x: clean_text(x)))
# data9 = pd.DataFrame(data9.Speeces.apply(lambda x: clean_text(x)))
# data10 = pd.DataFrame(data10.Speeces.apply(lambda x: clean_text(x)))



#remove stopwords

for i in range(len(data1)):
  data1.iloc[i]['Speeces'] = [word for word in data1.iloc[i]['Speeces'].split(' ') if word not in stopwords.stopwords(["da", "en"])]
  #print logs to monitor output 
  if i % 1000 == 0:
    sys.stdout.write('\rc = ' + str(i) + ' / ' + str(len(data1)))


data1_array=[value[0] for value in data1.iloc[0:].values] #list of lists, each column is csv is know described by words 




#%%
'''implementing LDA and choosing optimal number of topics k'''

id2word = gensim.corpora.Dictionary(data1_array)
corpus = [id2word.doc2bow(text) for text in data1_array]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model=ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("model with num_topics= ",num_topics)
        print('\n')

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data1_array, start=2, limit=40, step=1)

# Show graph
limit=40; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
#needed 10h22 minutes to run , optimal k=6
optimal_model_lda= model_list[5]

topics_optimal_model_lda=optimal_model_lda.print_topics()
pprint(topics_optimal_model_lda)
doc_optimal_model_lda = optimal_model_lda[corpus]
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(optimal_model_lda, corpus, id2word)
vis #
pyLDAvis.show(vis) #

