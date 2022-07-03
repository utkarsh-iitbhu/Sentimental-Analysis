import numpy as np
import pandas as pd
from string import punctuation
import re
import nltk
from nltk.corpus import twitter_samples
import random
# nltk.download('stopwords')
import string   
from tensorflow.keras.preprocessing.text import Tokenizer                        
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
# from nltk.tokenize import TweetTokenizer
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.tokenize import word_tokenize
# nltk.download("punkt")
from tokenizer import tokenizee

def tok(str):
    # print("str : "+str)
    maxlen = 200
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(str)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences([str])
    text = pad_sequences(sequences, maxlen=maxlen)
    # print("text: ")
    # print(text)
    return text

    
def stemming(word):
    stemmer = PorterStemmer() 
    list1=[]
    for i in word.split():
        list1.append(stemmer.stem(i))
    return ' '.join(list1)
    
def pro(tweets):

    l1 = {'clean_text':[tweets],'category':[0]}
    tweets= pd.DataFrame(l1)
    stopwords_english = stopwords.words('english') 
    tweets['clean_text'] = tweets['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_english)]))
    tweets['clean_text'] = tweets['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    stemmer = PorterStemmer() 
    def stemming(word):
        list1=[]
        for i in word.split():
            list1.append(stemmer.stem(i))
        return ' '.join(list1)
    tweets['clean_text'] = tweets['clean_text'].apply(lambda x:stemming(x)) 
    tweets['category'] = [2 if x == -1 else x for x in tweets['category']]
    ch = tweets.clean_text[0]
    import pickle
    maxlen = 200
    pickle_in = open("dict.pkl","rb")
    tt = pickle.load(pickle_in)
    sequences = tt.texts_to_sequences([ch])
    dew = pad_sequences(sequences, maxlen=maxlen)
    return dew

# str = "I am anti hindu, I am violent, rude, aggressive"
# str3 = "I am a good boy and , I love to play peacefully"

def pre_process(st):
    return pro(st)

# ans = pro(str3)
# # print(ans.shape)
# # print(ans)
# from keras.models import load_model
# model= load_model("model_lstm.h5")
# pr = model.predict(ans)

# # print(pr)
# a = pr[0][0]
# b = pr[0][1]
# c = pr[0][2]
# mx = max(pr[0][0],pr[0][1],pr[0][2])
# print(a,b,c)
# print(mx)
# if mx==a:
#     print("Neu")
# elif mx==b:
#     print("Pos")
# else:
#     print("Neg")
