import os
os.chdir("/home/aqumar/Downloads/subdir3_chatlogs")

for chats in range(30000,36895):
    filename = "chatlog" + str(chats) + ".csv"
    filename_out = "/home/aqumar/Downloads/clust/chatlog_" + str(chats) + ".csv"
    with open(filename) as fp, open(filename_out,'w') as fp1:
        for line in fp:
            fp1.write(line)
    fp.close()
    fp1.close()

open("all_chats.csv",'w').close()
os.chdir("/home/aqumar/Downloads/clust")
for chats in range(30000,36895):
    filename = "chatlog_" + str(chats) + ".csv"
    with open(filename) as fp, open("all_chats.csv",'a'):
        for line in fp:
            fp1.write(line)
    fp.close()
    fp1.close()
            

import pandas as pd

import numpy as np

import nltk

nltk.download()

doc = []
for chats in range(30000,36895):
    filename = "chatlog_" + str(chats) + ".csv"
    with open(filename) as fp:
        doc.append(fp.readlines())
    fp.close()
            


import collections

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from pprint import pprint


def word_tokenizer(text):
    
    tokens = word_tokenize(text)

    stemmer = PorterStemmer()

    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('german')]

    return tokens




def cluster_sentences(sentences, nb_of_clusters):
    

    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,

    stop_words=stopwords.words('german'),

    lowercase=True)

    #builds a tf-idf matrix for the sentences

    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    kmeans = KMeans(n_clusters=nb_of_clusters)

    kmeans.fit(tfidf_matrix)

    #cluster_errors.append( kmeans.inertia_ )

    clusters = collections.defaultdict(list)

    for i, label in enumerate(kmeans.labels_):
        
        clusters[label].append(i)

    return dict(clusters)




doc1 = pd.Series(doc)

for i in range(len(doc1)):
    doc1[i] = ' '.join(doc[i])

sentences = doc1

nclusters= 34

clusters = cluster_sentences(sentences, nclusters)

for cluster in range(nclusters):
    
    print(len(clusters[cluster]))

os.chdir("/home/aqumar/Downloads/kmeans")


for cluster in range(nclusters):
    
    filename = "cluster_" + str(cluster)

    open(filename,"w").close()

    for i,sentence in enumerate(clusters[cluster]):
        
        with open(filename, "a") as myfile:
            
            myfile.write(sentences[sentence] + "\n")
