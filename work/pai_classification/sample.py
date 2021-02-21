#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from PIL import Image
import glob
import os
import pickle

import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# In[3]:


# train_x = np.empty((0, 90,66,4))
train_x = np.empty((0, 90*66))
# train_x = np.array()
train_y = np.empty((0, 1))
files = glob.glob('pai-images/*')
for file in files:
    img = np.array(Image.open(file).convert('L'))
#     print(file, img.shape)
#     print(img)
#     if img.shape[0] != 90 and img.shape[1] != 66:
#         os.remove(file)
    fname = file[11:]
    x = fname.find('-')
    train_x = np.vstack((train_x, [img.flatten()]))
    train_y = np.vstack((train_y, [fname[0:x]]))
print('train_x.shape', train_x.shape)
# print('train_y.shape', train_y.shape)
train_y = pd.get_dummies(pd.DataFrame(train_y)).values
print('train_y.shape', train_y.shape)

print(train_x[0][0:10])

# 0~1に収める
scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

# train_x = train_x / 255.0

print(train_x[0][0:10])


# In[6]:


model = Sequential()
# Dense(64) は，64個のhidden unitを持つ全結合層です．
# 最初のlayerでは，想定する入力データshapeを指定する必要があり，ここでは20次元としてます．
model.add(Dense(5000, activation='relu', input_dim=90*66))
model.add(Dropout(0.2))
# model.add(Dense(5000, activation='relu', input_dim=90*66))
# model.add(Dropout(0.2))
# model.add(Dense(5000, activation='relu', input_dim=90*66))
# model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(37, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_x, train_y,
          epochs=50,
          batch_size=128)
score = model.evaluate(train_x, train_y, batch_size=128)


# In[ ]:





# In[ ]:





# In[ ]:




