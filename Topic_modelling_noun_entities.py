import spacy
import de_core_news_md
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

nlp = de_core_news_md.load()

number_clusters = range(0,34)
for i in number_clusters:
    cluster = str(i) + '.cl'
    fp = open(cluster,'r',encoding="utf8")
    df = fp.readlines()
    fp.close()
    all_noun = []
    for sent in range(len(df)):
        sentence = df[sent]
        doc = nlp(sentence)
        noun_list = [w.text for w in doc if w.pos_ in 'NOUN']
        if len(noun_list) == 0:
            continue
        all_noun.append(noun_list)
    doc_complete = all_noun

    stop1 = set(stopwords.words('german'))
    stop2 = set(stopwords.words('english'))
    stop3 = set(['customer','agent','name'])
    stop = stop1.union(stop2)
    stop = stop.union(stop3)
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized
  
    flat_list = [item for sublist in doc_complete for item in sublist]
    doc_complete = pd.Series(flat_list)
    doc_clean = [clean(doc).split() for doc in doc_complete]
    
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=15, id2word = dictionary, passes=50)
    
    res = ldamodel.print_topics(num_topics=15, num_words=5)
    
    filename = 'topics_cluster_' + str(i)
    f = open(filename,'w')
    for t in res:
        f.write(' '.join(str(s) for s in t) + '\n')
    f.close()
    
    
    
