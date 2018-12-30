"""
Author : Swapnil Gaikwad
Title : sentiment classification in 2 classes (binary classifier) => 0 - non hate speech, 1 - hate speech
tools : nltk, sklearn, numpy
dataset : training data size - 31962, test data size - 17197
features used => tfidf
"""

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re, os, traceback, json, string, warnings
from collections import defaultdict
from many_stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords

import scipy
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score


class SentimentAnalyzer(object):
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.many_stop_words = list(get_stop_words('en'))
        self.stopwords = self.many_stop_words
        self.labels = ['negative','positive']
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, max_df=0.2)
        self.count_vect = CountVectorizer()
        self.clf = LinearSVC()


    def clean_str(self, string): # unused
        string = re.sub(r"[^A-Za-z0-9]", " ", string)
        string = re.sub(r"\'s", " ", string)
        string = re.sub(r"\'ve", " ", string)
        string = re.sub(r"n\'t", " ", string)
        string = re.sub(r"\'re", " ", string)
        string = re.sub(r"\'d", " ", string)
        string = re.sub(r"\'ll", " ", string)
        string = re.sub(r"#", " ", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"!", " ", string)
        string = re.sub(r"\(", " ", string)
        string = re.sub(r"\)", " ", string)
        string = re.sub(r"\?", " ", string)
        string = re.sub(r"\s{3,}", " ", string)
        sring = self.removeStopWords(string)
        string = self.getLemma(string)

        return string.strip()

    def removeStopWords(self, text):
        return " ".join([token for token in text.split() if token not in self.stopwords])

    def checkLemma(self, wrd):
        return nltk.stem.WordNetLemmatizer().lemmatize(nltk.stem.WordNetLemmatizer().lemmatize(wrd, 'v'), 'n')

    def getLemma(self, text):
        text_list = []
        text_list = [self.checkLemma(tok) for tok in text.lower().split()]
        text = " ".join(text_list)
        return text

    def readData(self):
        # step 1 === read dataset
        train  = pd.read_csv('train_E6oV3lV.csv')
        test = pd.read_csv('test_tweets_anuFYb8.csv')
        print('\n train size == ', len(train), train.keys())
        print("\n summary : \n",train.head())
        print "\n size of test --- ",len(test)
        train['tweet'] = [self.clean_str(eg) for eg in train['tweet']]
        test['tweet'] = [self.clean_str(eg) for eg in test['tweet']]

        return train, test


    def trainModel(self):

        #################### step 1 - read dataset
        train, test = self.readData()

        #################### step 2 - data analysis
        toxic_train = [lab for lab in train['label'] if lab == 1] 
        print "\n no. of toxic tweets in train --- ",len(toxic_train)
        toxic_tweets = []
        for ind, series in train.iterrows():
            if series['label'] == 1:
                toxic_tweets.append(self.clean_str(series['tweet']))

        toxic_tweets_vector = self.count_vect.fit_transform(toxic_tweets)
        print "\n toxic tweets top words --- ",self.count_vect.get_feature_names()[:500]
        vocab = self.count_vect.vocabulary_
        print "\n toxic tweets top words --- ",type(vocab), sorted(vocab, reverse=True)

        #################### step 3 - convert text to feaures using TF-IDF
        train_vector = self.vectorizer.fit_transform(train['tweet'])
        test_vector = self.vectorizer.transform(test['tweet'])
        train_label = train['label'].ravel()
        print "\n train_vector --- ",train_vector.shape, type(train_vector)        

        #################### step 4 - train model / classifier
        self.clf.fit(train_vector, train_label)
        print "\n training accuracy :: ",self.clf.score(train_vector, train_label)

        #################### step 5 - predict using trained model
        test_pred = self.clf.predict(test_vector)

        ##################### step 6 - write predictions in csv format for submission
        self.writeToCsv(test, test_pred)
        

    def writeToCsv(self, test, test_labels):
        try:
            submission = defaultdict(list)
            submission['id'].extend(test['id'])
            submission['label'].extend(test_labels)
            submission = pd.DataFrame(submission)
            submission.to_csv( 'submission_svm.csv', index=False)
            print("\n saved results in csv successfully === ")
        
        except Exception as e:
            print("\n error ", e, "\n traceback === ",traceback.format_exc())
        
    
if __name__ == '__main__':
    obj = SentimentAnalyzer()
    obj.trainModel()