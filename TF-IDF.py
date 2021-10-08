#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:41:41 2021

@author: kshama.singh
"""


import nltk
               
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


paragraph = open("../TF-IDF/TF-IDF.txt", 'r').read()
#print(paragraph)
              
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
tfidf = cv.fit_transform(corpus).toarray()