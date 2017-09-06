#Creating the tf-idf matrix for the queries.


import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import ast
import numpy as np

fp = open('all_queries','r')

all_queries = fp.readlines()[1:]


all_queries = pd.Series(all_queries)

import collections

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer




def word_tokenizer(text):
    
    tokens = word_tokenize(text)

    stemmer = PorterStemmer()

    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('german')]

    return tokens


tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,

    stop_words=stopwords.words('german'),
                                   
    lowercase=True)


sentences = list(all_queries)

#builds a tf-idf matrix for the sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

#Building the cosine similarity matrix between every query.

lst = [ast.literal_eval(str(linear_kernel(tfidf_matrix[i], tfidf_matrix[j]).flatten()))[0] for i in range(tfidf_matrix.shape[0]) for j in range(tfidf_matrix.shape[0])]


#Storing the similarity matrix.

with open("output_tfidf_cosine", "wb") as fp:
    pickle.dump(lst,fp)

fp.close()

#Loading the similarity matrix.

import pickle
with open("output_tfidf_cosine", "rb") as fp: 
    lst = pickle.load(fp)
fp.close()

#Reshaping the similarity matrix.

cosine_matrix = np.array(lst).reshape(tfidf_matrix.shape[0],tfidf_matrix.shape[0])


df = pd.DataFrame(cosine_matrix)

#for each data point p = find array of size k which stores the nearest neighbor points
#point = query for us. query is represented as a tf-idf vector
#similarity between queries (p, q) = distance_metric on tf-idf of p and q
#distance metric = cosine

dict_top_5 = {}
for i in range(tfidf_matrix.shape[0]):
    each_row = list(df.loc[i])
    index_top_5 = sorted(range(len(each_row)), key=lambda i: each_row[i], reverse=True)[:5]
    dict_top_5[i] = index_top_5
    


#for each data point q = find all points p where q occurs in the kNN set of p
#this array can be of different sizes.. depends on the how well important the query is
#end of this submodule, each data point q will have its own set of points which consider q to be one of their kNN

rnn_set = {}
for k in range(tfidf_matrix.shape[0]):
    rnn_set_sublist = []
    for i in range(tfidf_matrix.shape[0]):
        if k in dict_top_5[i]:
            rnn_set_sublist.append(i)
    rnn_set[k] = rnn_set_sublist
    

#for each data point q, if the size of kRNN set is less than (k)
#label this point as potential noise. keep it aside
#this point is not suitable to find clusters

rnn_set_imp = {}
rnn_set_non_imp = {}

for i in range(tfidf_matrix.shape[0]):
    if len(rnn_set[i]) < 3:
        rnn_set_non_imp[i] = rnn_set[i]
    else:
        rnn_set_imp[i] = rnn_set[i]
        

#start from first point, take its kRNN set.
#for each point in kRNN set, find their kRNN sets
#(breadth-first traversal) to identify kRNN links
#keep growing this cluster
#once there are no more to add, label all points identified so far as Cluster 1
#take the next set of unlabelled important points
#do the same..
#cover till all important points are labelled

all_cluster = []
labelled_points = []

def create_cluster(start_point):
    cluster = []
    listtocheck = [start_point]
    for j in listtocheck:
        if j not in labelled_points:
            labelled_points.append(j)
            cluster.append(j)
            if j in rnn_set_imp:
                listtocheck += rnn_set_imp[j]
    return cluster
        


for i in rnn_set_imp.keys():
    if i not in labelled_points:
        cluster_res = create_cluster(i)
        all_cluster.append(cluster_res)
        
    
    
    
    
#Writing the results.


cluster_no = 0
for each_cluster in all_cluster:
    filename = "KNN_cluster_result/cluster_" + str(cluster_no)
    open(filename,'w').close()
    with open(filename,'a') as fp:
        for cluster_index in each_cluster:
            fp.write(sentences[cluster_index])
            fp.write("\n")
    fp.close()
    cluster_no += 1
        


#for each non-important point, if its kRNN set is more than 3, find if there’s a majority cluster the points in kRNN set belong to. If there is, assign this point to that cluster

#Submodule 5 - needs to be done only if clusters are skeletal in nature.

#Needs to be modify.

#for i in rnn_set_non_imp.keys():
#    if len(rnn_set_non_imp[i]) > 3:
#        cluster_map = {}
#        cluster_no = 1
#        for each_cluster in all_cluster:
#            cluster_map[cluster_no] = len(list(set(each_cluster) & set(rnn_set_non_imp[i])))
#            cluster_no += 1
#        lst = [k for k, v in cluster_map.items() if v == max(cluster_map.values())]
#        cluster_tobelong = lst[0]
#        if rnn_set_non_imp[i] not in all_cluster[cluster_tobelong]:
#            all_cluster[cluster_tobelong] += rnn_set_non_imp[i]
#            print("Added")


        
            
