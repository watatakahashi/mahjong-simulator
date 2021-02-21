#!/usr/bin/env python
# coding: utf-8

# In[24]:


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


# In[25]:


# train_x = np.empty((0, 90,66,4))
train_x = np.empty((0, 90*66))
# train_x = np.array()
train_y = np.empty((0, 1))
files = glob.glob('jantama-pai-images/*')
for file in files:
    img = np.array(Image.open(file).convert('L'))
    print(file, img.shape)
    img.resize((128, 128))
    print(file, img.shape)
    print(img)
    fname = file[file.find('/')+1:].split(".png")[0]
    fname = fname.replace('0m', '0').replace('1m', '1').replace('2m', '2').replace('3m', '3').replace('4m', '4')    .replace('5m', '5').replace('6m', '6').replace('7m', '7').replace('8m', '8').replace('9m', '9')    .replace('0p', '10').replace('1p', '11').replace('2p', '12').replace('3p', '13').replace('4p', '14')    .replace('5p', '15').replace('6p', '16').replace('7p', '17').replace('8p', '18').replace('9p', '19')    .replace('0s', '20').replace('1s', '21').replace('2s', '22').replace('3s', '23').replace('4s', '24')    .replace('5s', '25').replace('6s', '26').replace('7s', '27').replace('8s', '28').replace('9s', '29')    .replace('1z', '30').replace('2z', '31').replace('3z', '32').replace('4z', '33').replace('5z', '34')    .replace('6z', '35').replace('7z', '36')
    print(fname)
#     x = fname.find('-')
#     train_x = np.vstack((train_x, [img.flatten()]))
#     train_y = np.vstack((train_y, [fname[0:x]]))
# print('train_x.shape', train_x.shape)
# print('train_y.shape', train_y.shape)
# train_y = pd.get_dummies(pd.DataFrame(train_y)).values
# print('train_y.shape', train_y.shape)


# In[ ]:





# In[ ]:




