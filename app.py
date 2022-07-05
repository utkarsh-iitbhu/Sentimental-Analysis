import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
# import nltk
from keras.models import load_model
# from tenacity import retry
# from tensorflow.keras.preprocessing.text import Tokenizer                        
# from nltk.corpus import stopwords 
# from nltk.stem import PorterStemmer
# import numpy as np
import os
from process import pre_process, pro
model= load_model("model_lstm.h5")
# import numpy as np
# import spacy
# from sklearn.svm import LinearSVC
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# import joblib
# import string
# from tokenizer import tokenizee
# We have loaded the modek
# os.chdir(r'C:\Users\utkar\Desktop\Hate speech\Twitter_kagle')
# model = load_model("network_lstm.h5")
# model = load_model("C:\Users\utkar\Desktop\Hate speech\Twitter_kagle\model_lstm.h5")
# Now we have to process the given input and predict it using the trained model
# model_load.predict(X_val)





stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)


# Define predict function
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/',methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':
        text = request.form['tweet']
        # def pre_process(text):
        #     return pro(text)
        lis = pre_process(text)
        pred = model.predict(lis)
        a = pred[0][0]
        b = pred[0][1]
        c = pred[0][2]
        mx = 0 
        if a>b and a>c:
            mx = a
        elif b>a and b>c:
            mx = b
        elif c>a and c>b:
            mx = c
        # mx = max(pred[0][0],pred[0][1],pred[0][2])
        
        # 1 positive 2 negative 0 neutral
        if (mx==b):
            return render_template('index.html', prediction_textb='Positive'+' '+str(b))
        elif (mx==a):
            return render_template('index.html', prediction_texta='Neutral'+' '+str(a))
        elif (mx==c):
            return render_template('index.html', prediction_textc='Negative'+' '+str(c))
        # else:
        #     return render_template('index.html', prediction_text='Cannot classify'+' '+str(a) +' '+str(b) +' '+str(c))
    # else:
    #     return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
