#!/usr/bin/env python
# coding: utf-8

# In[32]:


#İlk çalıştırmada alınan hatalar ikinci seferde düzelmektedir. Lütfen iki sefer çalıştırınız.
import re
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D

from keras.utils.np_utils import to_categorical

import warnings
warnings.filterwarnings("ignore")


# In[33]:


Dataset = pd.read_csv('Sentiment.csv')


# In[34]:


Dataset.head()


# In[35]:


m, n = Dataset.shape

print(f'Number of rows in dataset : {m}')
print(f'Number of columns in dataset : {n}')


# In[36]:


Dataset.columns


# In[37]:


# Gerekli sütunlar seçildi.

Dataset = Dataset[['text', 'sentiment']]


# In[38]:


Dataset.head()


# In[39]:


Dataset['text']


# In[40]:


Dataset['text'] = Dataset['text'].map(lambda x: x.lower())
Dataset['text'] = Dataset['text'].map(lambda x: re.sub('[^a-z0-9\s]','', x))

for i in range(Dataset.shape[0]):
    Dataset['text'][i] = Dataset['text'][i].replace('rt', '')


# In[41]:


Dataset['text']


# In[42]:


# Duygu sütununda benzersiz değerlerin sayısı gösterildi.

Uniq_vals = Dataset['sentiment'].value_counts()
Uniq_vals


# In[43]:


#Duyguların yüzdelik değerleri bulunarak görselleştirme işlemi yapıldı.
LABELS = ['Negative', 'Neutral', 'Positive']

def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

fig, ax = plt.subplots(figsize =(8, 6))
ax.pie(Uniq_vals, 
       labels=LABELS, 
       autopct=lambda pct: func(pct, Uniq_vals))

ax.legend(loc ="center left", bbox_to_anchor =(1, 0, 0.5, 1))
ax.set_title('Distribution of Sentiment')
plt.show()


# In[44]:


max_features = 2000   

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(Dataset['text'].values)
X = tokenizer.texts_to_sequences(Dataset['text'].values)
X = pad_sequences(X)


# In[45]:


X


# In[46]:


#İkili değişkenlere dönüştürelim.
y = pd.get_dummies(Dataset['sentiment']).values 


# In[47]:


y


# In[48]:


lstm_out = 196

# Sıralı model tanımlandı.
model = Sequential()

# 128 boyutlu vektöre gömme işlemi yapıldı.
model.add(Embedding(max_features, 128, input_length=X.shape[1]))

# Aşırı öğrenme azaltaldı. 
model.add(Dropout(0.2))

# LSTM Katmanı eklendi.
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2)))

# Yoğun katmanı eklendi.
model.add(Dense(3, activation='softmax')) 


# In[49]:


#Model optimizasyon işlemi yapıldı.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[50]:


model.summary()


# In[51]:


#Veriler eğitim ve test olarak ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=44, 
                                                    shuffle=True)


# In[52]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[53]:


#Model eğitildi.
BATCH_SIZE = 64
EPOCHS = 8

hist = model.fit(X_train, y_train, 
                 epochs = EPOCHS, 
                 validation_data = (X_test, y_test), 
                 batch_size = BATCH_SIZE, 
                 verbose = 1)


# In[54]:


score, acc = model.evaluate(X_test, y_test, batch_size = BATCH_SIZE, verbose=0)

print('Score: {:.2f}'.format(score))
print('Accuracy: {:.2f}'.format(acc))


# In[55]:


# Modelin test işlemi yapıldı. 

tweet = ["I'm happy."]

# Önceden kullanılan tokenizer örneğiyle vektörleştirildi.
tweet = tokenizer.texts_to_sequences(tweet)

tweet = pad_sequences(tweet, maxlen=28, dtype='int32', value=0)

sentiment = model.predict(tweet, batch_size=1, verbose=1)[0]

if(np.argmax(sentiment) == 0):
    print("Negative")
    
elif (np.argmax(sentiment) == 1):
    print("Positive")
    
elif (np.argmax(sentiment) == 2):
    print("Neutral")


# In[58]:


#Model kaydedildi.
from keras.models import load_model
model.save('YZ_Proje.h5')


# In[61]:


#Uygulamanın arayüzü oluşturuldu.
from tkinter import messagebox, ttk, Entry, Label, Tk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd

class SentimentAnalysisApp:
    def __init__(self, model_path):
        self.model_path = model_path
        self.max_features = 2000
        self.load_model_and_tokenizer()

        self.root = Tk()
        self.root.title("Sentiment Analysis App")
        self.root.geometry("500x300")

        self.create_widgets()
        self.root.mainloop()

    def load_model_and_tokenizer(self):
        self.model = load_model(self.model_path)
        self.tokenizer = Tokenizer(num_words=self.max_features, split=' ')
        # Dataset yerine uygun bir veri kullanılmalı
        self.tokenizer.fit_on_texts(Dataset['text'].values)

    def create_widgets(self):
        title_label = Label(self.root, text="Sentiment Analysis App", font=("Helvetica", 20, "bold"), background="#202da1", foreground="white")
        title_label.pack(pady=10, fill='x')

        self.text_entry = Entry(self.root, width=50, font=("Helvetica", 14), background="#ECF0F1")
        self.text_entry.pack(pady=10)

        analyze_button = ttk.Button(self.root, text="Analyze", command=self.analyze_sentiment, style='TButton')
        analyze_button.pack(pady=10)

    def analyze_sentiment(self):
        tweet = self.text_entry.get()

        tweet_vectorized = self.tokenizer.texts_to_sequences([tweet])
        tweet_padded = pad_sequences(tweet_vectorized, maxlen=28, dtype='int32', value=0)

        sentiment = self.model.predict(tweet_padded, batch_size=1, verbose=0)[0]

        sentiments = ['Negative 😢', 'Positive 😊', 'Neutral 🤠']
        
        # Duygu tespiti yapma ve eşik değerleri yerine oranları kullanma
        positive_threshold = 0.4  # Pozitif olma olasılığı için eşik değer
        negative_threshold = 0.4  # Negatif olma olasılığı için eşik değer

        if sentiment[1] > positive_threshold:
            result_text = sentiments[1]  # Positive
        elif sentiment[0] > negative_threshold:
            result_text = sentiments[0]  # Negative
        else:
            result_text = sentiments[2]  # Neutral

        messagebox.showinfo("Sentiment Result", f"The sentiment of the tweet is: {result_text}")

if __name__ == "__main__":
    app = SentimentAnalysisApp(model_path="YZ_Proje.h5")


# In[ ]:





# In[ ]:





# In[ ]:




