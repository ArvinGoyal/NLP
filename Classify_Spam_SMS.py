# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:46:06 2019

@author: arvgoyal
"""
import nltk
import os
import pandas as pd

os.chdir ("D:\Arvin\ArtInt\\NLP\Text Mining")   
sms = pd.read_csv ("sms_spam.csv", header = 0)

########################################
#tokenizing the data
########################################
from nltk import word_tokenize as w_tkn
sms['token'] = sms['text'].apply(w_tkn)

#####################################
# Stopword: identifying and removing stop words
#####################################
from nltk.corpus import stopwords as stpw

StWords = set (stpw.words('english'))

def removeStWords(words):
    flt_words = []
    for w in words:
        if w not in StWords:
            flt_words.append(w)
           
    return flt_words

sms ['prm_token'] = sms['token'].apply(removeStWords)

########################################
#Stemming the data
########################################
from nltk.stem import PorterStemmer

ps = PorterStemmer()
def Stemmer(words):
    stemmedWord = []
    for w in words:
        stemmedWord.append(ps.stem(w))
    
    return stemmedWord

sms ['stm_token'] = sms['prm_token'].apply(Stemmer)

########################################
#Creating proccesed sentances
########################################
def createFilterdSentence(words):
    str1 = ''
    str1 =  ' '.join(words)
    return str1

sms['newText'] = sms['stm_token'].apply(createFilterdSentence)

sms['text']=sms['newText']
#del sms['token']
del sms['prm_token']
del sms['stm_token']
del sms['newText']

########################################
#Split Tset and Train
########################################
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# We need target feature also in test for NLTK accuracy hence doing this split manualy
sms_train = sms[:4559] 
sms_tst  = sms[4559:]

########################################
#Creating Bag of words and fetures from Training Data only
########################################
bag_of_words=[]
bag_of_words = ' '.join(sms_train.iloc[:,1])

bag_of_words = w_tkn(bag_of_words)

bag_of_words = nltk.FreqDist(bag_of_words)
type(bag_of_words)
sorted_bag = bag_of_words.most_common()

print (sorted_bag[:50])

features = []
for i in range(21, 3000):
    features.append(sorted_bag[i][0])

############################################
#tokenize the Train and test sets as I have deleted previous tokens
############################################
#sms_train['token'] = sms_train['text'].apply(w_tkn)
#sms_tst['token'] = sms_tst['text'].apply(w_tkn)

############################################
# Creating DTM (Documnet Term Matrix) for every sentance in Train and Test set
############################################
def find_features(list_of_tokens):
    words = set(list_of_tokens)
    sms_features = {}
    for w in features:
        sms_features[w] = (w in words)    ## it will set true or false
        
    return sms_features 
 
sms_train['features']    = sms_train['token'].apply(find_features)
sms_tst['features']    = sms_tst['token'].apply(find_features)

del sms_train['token']
del sms_tst['token']

del sms_train['text']
del sms_tst['text']

sms_train.info()

############################################
# Creating a new tuple column to have DTM and its categoury within
############################################
def createLabeledFeatures(obs):
    
    return (obs['features'],obs['type'])

sms_train['featureset'] = sms_train.apply(createLabeledFeatures, axis =1 )
sms_tst['featureset'] = sms_tst.apply(createLabeledFeatures, axis =1 )
    
print(sms_train ['featureset'][0])



############################################
# training the model on sms_train
############################################
classifier = nltk.classify.NaiveBayesClassifier.train(sms_train['featureset'])
############################################
# testing the accuracy of model on sms_tst
############################################
print("NB Classifier Accuracy", nltk.classify.accuracy(classifier,sms_tst['featureset']))







