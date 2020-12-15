#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''LDA on 2009 data '''

import numpy as np
import pandas as pd
import re, nltk, gensim , spacy, string

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

#from nltk.corpus import stopwords
#stop_words = stopwords.words('danish')

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
from gensim.models import CoherenceModel
#---------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import stopwordsiso as stopwords
import sys
#from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF,PCA;
from sklearn.preprocessing import normalize;
import pickle;

import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from tqdm import tqdm
import random

data1 = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case 2/new_data/20091.csv', error_bad_lines=False);
data1 = pd.read_csv('new_datasets/20091.csv', error_bad_lines=False);
data2 = pd.read_csv('new_datasets/20101.csv', error_bad_lines=False);
data3 = pd.read_csv('new_datasets/20102.csv', error_bad_lines=False);
data4 = pd.read_csv('new_datasets/20111.csv', error_bad_lines=False);
data5 = pd.read_csv('new_datasets/20121.csv', error_bad_lines=False);
data6 = pd.read_csv('new_datasets/20131.csv', error_bad_lines=False);
data7 = pd.read_csv('new_datasets/20141.csv', error_bad_lines=False);
data8 = pd.read_csv('new_datasets/20142.csv', error_bad_lines=False);
data9 = pd.read_csv('new_datasets/20151.csv', error_bad_lines=False);
data10 = pd.read_csv('new_datasets/20161.csv', error_bad_lines=False);

data1 = data1[['Speeces']].astype('str')
data2 = data2[['Speeces']].astype('str')
data3 = data3[['Speeces']].astype('str')
data4 = data4[['Speeces']].astype('str')
data5 = data5[['Speeces']].astype('str')
data6 = data6[['Speeces']].astype('str')
data7 = data7[['Speeces']].astype('str')
data8 = data8[['Speeces']].astype('str')
data9 = data9[['Speeces']].astype('str')
data10 = data10[['Speeces']].astype('str')

data_text = [data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]
#data_text=[data1]
#----------------------------------------------------------------
all_data = []
for data in data_text:
  for i in range(len(data)):
    data.iloc[i]['Speeces'] = [word for word in data.iloc[i]['Speeces'].split(' ') if word not in stopwords.stopwords(["da", "en"])];
    #print logs to monitor output 
    if i % 1000 == 0:
      sys.stdout.write('\rc = ' + str(i) + ' / ' + str(len(data)));

  all_data.append(data)
  
#------------------------------------------------------------------

#get the words as an array for lda input
#data_array = np.zeros((109, 10))
data_array = []
#data_text1 = pd.DataFrame(data_text1)
#data_array = [value[0] for value in data_text1.iloc[0:].values]
for data in all_data:
  data_array.append([value[0] for value in data.iloc[0:].values])
print(np.shape(data_array[0]))

#----------------------------------------------------------------

#NMF Implementation

#the count vectorizer module needs string inputs, not array, so I join them with a space. This is a very quick operation.

data_array_sentences = []
for data in data_array:
  data_array_sentences.append([' '.join(text) for text in data])

#data_array_sentences = [' '.join(text) for text in data_array]
  
#----------------------------------------------------------------
  
vectorizer = CountVectorizer(analyzer='word', max_features=5000);

count_array = []
for data in data_array_sentences:
  count_array.append(vectorizer.fit_transform(data))

#count_array = vectorizer.fit_transform(data_array_sentences)
  
#----------------------------------------------------------------
  
transformer = TfidfTransformer(smooth_idf=False)

array_transform = []
for counts in count_array:
  array_transform.append(transformer.fit_transform(counts))

#array_tfidf = transformer.fit_transform(count_array)
  
#----------------------------------------------------------------  

array_norm = []
for ar in array_transform:
  array_norm.append(normalize(ar, norm='l1', axis=1))

#xtfidf_norm = normalize(array_tfidf, norm='l1', axis=1)
  
#----------------------------------------------------------------
  
def get_nmf_clusters(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(num_clusters):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-n_top_words- 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

#----------------------------------------------------------------
    
#obtain a NMF model.
num_clusters = 10
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[0])
nmf_20091 = get_nmf_clusters(model, top)
nmf_20091.to_csv('nmf_20091.csv', encoding='utf-8')

model.fit(array_norm[1])
nmf_20101 = get_nmf_clusters(model, top)
nmf_20101.to_csv('nmf_20101.csv', encoding='utf-8')

model.fit(array_norm[2])
nmf_20102 = get_nmf_clusters(model, top)
nmf_20102.to_csv('nmf_20102.csv', encoding='utf-8')

model.fit(array_norm[3])
nmf_20111 = get_nmf_clusters(model, top)
nmf_20111.to_csv('nmf_20111.csv', encoding='utf-8')

model.fit(array_norm[4])
nmf_20121 = get_nmf_clusters(model, top)
nmf_20121.to_csv('nmf_20121.csv', encoding='utf-8')

model.fit(array_norm[5])
nmf_20131 = get_nmf_clusters(model, top)
nmf_20131.to_csv('nmf_201031.csv', encoding='utf-8')

model.fit(array_norm[6])
nmf_20141 = get_nmf_clusters(model, top)
nmf_20141.to_csv('nmf_20141.csv', encoding='utf-8')

model.fit(array_norm[7])
nmf_20142 = get_nmf_clusters(model, top)
nmf_20142.to_csv('nmf_20142.csv', encoding='utf-8')

model.fit(array_norm[8])
nmf_20151 = get_nmf_clusters(model, top)
nmf_20151.to_csv('nmf_20151.csv', encoding='utf-8')

model.fit(array_norm[9])
nmf_20161 = get_nmf_clusters(model, top)
nmf_20161.to_csv('nmf_20161.csv', encoding='utf-8')
#-----------------------------------------------------------------

#LDA Implementation
num_topics = 6

lda = []
for data in data_array:
  #data_array_Id2word.append(gensim.corpora.Dictionary(data))
  id2word = gensim.corpora.Dictionary(data);
  corpus = [id2word.doc2bow(text) for text in data]
  lda.append(ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics))
#-----------------------------------------------------------------

def get_lda_topics(model, num_topics):
    #word_dict = {};
    new_words = []
    for i in range(num_topics):
        #words = model.show_topic(i, topn = 10);
        new_words.append(model.show_topic(i, topn = 10));
        #word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    #return pd.DataFrame(word_dict);
    return pd.DataFrame(new_words);

#-----------------------------------------------------------------

lda_20091 = get_lda_topics(lda[0], num_topics).T
lda_20091.to_csv('lda_20091.csv', encoding='utf-8')

lda_20101 = get_lda_topics(lda[1], num_topics)
lda_20101.to_csv('lda_20101.csv', encoding='utf-8')

lda_20102 = get_lda_topics(lda[2], num_topics)
lda_20102.to_csv('lda_20102.csv', encoding='utf-8')

lda_20111 = get_lda_topics(lda[3], num_topics)
lda_20111.to_csv('lda_20111.csv', encoding='utf-8')

lda_20121 = get_lda_topics(lda[4], num_topics)
lda_20121.to_csv('lda_20121.csv', encoding='utf-8')

lda_20131 = get_lda_topics(lda[5], num_topics)
lda_20131.to_csv('lda_20131.csv', encoding='utf-8')

lda_20141 = get_lda_topics(lda[6], num_topics)
lda_20141.to_csv('lda_20141.csv', encoding='utf-8')

lda_20142 = get_lda_topics(lda[7], num_topics)
lda_20142.to_csv('lda_20142.csv', encoding='utf-8')

lda_20151 = get_lda_topics(lda[8], num_topics)
lda_20151.to_csv('lda_20151.csv', encoding='utf-8')

lda_20161 = get_lda_topics(lda[9], num_topics)
lda_20161.to_csv('lda_20161.csv', encoding='utf-8')
#------------------------------------------------------------------

#Optimal cases----------------------------------------------------

#NMF model.
#20091 dataset
num_clusters = 6
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[0])
nmf_20091 = get_nmf_clusters(model, top)

#LDA model
num_topics = 6

lda = []

id2word = gensim.corpora.Dictionary(data_array[0]);
corpus = [id2word.doc2bow(text) for text in data_array[0]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20091 = get_lda_topics(lda, num_topics).T
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
vis #'''did not display work with jupyter'''
pyLDAvis.show(vis) #this worked to display

#-------------------------------------------------------------------

#NMF model.
#20101 dataset
num_clusters = 6
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[1])
nmf_20101 = get_nmf_clusters(model, top)

#LDA model
num_topics = 6

lda = []

id2word = gensim.corpora.Dictionary(data_array[1]);
corpus = [id2word.doc2bow(text) for text in data_array[1]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20101 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display


#-------------------------------------------------------------------

#NMF model.
#20111 dataset
num_clusters = 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[3])
nmf_20111 = get_nmf_clusters(model, top)

#LDA model
num_topics = 4

lda = []

id2word = gensim.corpora.Dictionary(data_array[3]);
corpus = [id2word.doc2bow(text) for text in data_array[3]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20111 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display


#-------------------------------------------------------------------

#NMF model.
#20121 dataset
num_clusters = 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[4])
nmf_20121 = get_nmf_clusters(model, top)

#LDA model
num_topics = 6

lda = []

id2word = gensim.corpora.Dictionary(data_array[4]);
corpus = [id2word.doc2bow(text) for text in data_array[4]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20121 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display

#-------------------------------------------------------------------

#NMF model.
#20131 dataset
num_clusters = 4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[5])
nmf_20131 = get_nmf_clusters(model, top)

#LDA model
num_topics = 4

lda = []

id2word = gensim.corpora.Dictionary(data_array[5]);
corpus = [id2word.doc2bow(text) for text in data_array[5]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20131 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display

#-------------------------------------------------------------------

#NMF model.
#20141 dataset
num_clusters = 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[6])
nmf_20141 = get_nmf_clusters(model, top)

#LDA model
num_topics = 2

lda = []

id2word = gensim.corpora.Dictionary(data_array[6]);
corpus = [id2word.doc2bow(text) for text in data_array[6]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20141 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display

#-------------------------------------------------------------------

#NMF model.
#20151 dataset
num_clusters = 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[3])
nmf_20151 = get_nmf_clusters(model, top)

#LDA model
num_topics = 2

lda = []

id2word = gensim.corpora.Dictionary(data_array[8]);
corpus = [id2word.doc2bow(text) for text in data_array[8]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20151 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display

#-------------------------------------------------------------------

#NMF model.
#20121 dataset
num_clusters = 6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
model = NMF(n_components=num_clusters, init='nndsvd');

#for data in array_norm: 
#  model.fit(data)

top = 10
model.fit(array_norm[3])
nmf_20161 = get_nmf_clusters(model, top)

#LDA model
num_topics = 6

lda = []

id2word = gensim.corpora.Dictionary(data_array[9]);
corpus = [id2word.doc2bow(text) for text in data_array[9]]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
lda_20161 = get_lda_topics(lda, num_topics).T

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
# vis #'''did not display work with jupyter'''
# pyLDAvis.show(vis) #this worked to display



