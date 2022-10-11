#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import cifar10 


# In[12]:


import numpy
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils 


# In[13]:


#load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[14]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[15]:


#normalize
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32') 
X_train = X_train / 255.0 
X_test = X_test / 255.0


# In[16]:


y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]


# In[17]:


model = Sequential()


# In[18]:


model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(Dropout(0.2)) 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Flatten()) 
model.add(Dropout(0.2)) 
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation='softmax'))


# In[19]:


model.summary()


# In[20]:


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[21]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# In[ ]:


from keras.models import load_model 
model.save('D:\DL 1\CIFAR.h5')


# In[ ]:


model = load_model('D:\DL 1\CIFAR.h5')

