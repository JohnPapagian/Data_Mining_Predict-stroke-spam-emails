import pandas as pd
import nltk
from tensorflow.python.keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K

df = pd.read_csv(r'C:\Users\johnd\ΜΑΘΗΜΑΤΑ\ΕΞΟΡΥΞΗ ΔΕΔΟΜΕΝΩΝ\Πρότζεκτ\spam_or_not_spam\spam_or_not_spam.csv')

df1 = df.drop([1466]) #row 1466 is NaN and cause problems in tokenization

df1["tokens1"] = df1["email"].apply(nltk.word_tokenize) #with punctuation
tokenizer = RegexpTokenizer(r'\w+') #do not keep any type of symbol
df1['tokens'] = df1["email"].apply(tokenizer.tokenize) #without punctuation

stemmer = PorterStemmer()
df1['stemmed'] = df1['tokens'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.

df1['without_stopwords'] = df1['stemmed'].apply(lambda x: [item.lower() for item in x if item.lower() not in stop]) #remove stopwords

tf = TfidfVectorizer(lowercase='false', analyzer='word')
st = df1['without_stopwords'].apply(lambda x: ' '.join(x))

re = tf.fit_transform(st)
response = tf.fit_transform(st).toarray()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
X = response
Y = df1['label'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(18, activation='sigmoid', input_dim=17477))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])

history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test,Y_test),verbose=1)

model.summary()
