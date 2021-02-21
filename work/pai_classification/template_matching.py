#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img = cv2.imread("sample/lt.jpeg")
temp = cv2.imread("sample/5m2.jpeg")


# In[3]:


img.shape


# In[4]:


temp.shape


# In[5]:


# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# テンプレート画像の高さ・幅
h, w = temp.shape

# テンプレートマッチング（OpenCVで実装）
match = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF)
min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
pt = min_pt

# テンプレートマッチングの結果を出力
cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
cv2.imwrite("sample/res2.png", img)


# In[6]:


min_pt


# In[7]:


min_value


# In[8]:


# import pyautogui

# screenshot = pyautogui.screenshot()
# screenshot.save('スクリーンショット.png')


# In[9]:


import pyautogui
pyautogui.screenshot('filename.png')


# In[ ]:




