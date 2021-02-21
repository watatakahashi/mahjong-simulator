#!/usr/bin/env python
# coding: utf-8

# # 対面の河の画像認識

# In[ ]:


import pyautogui
from PIL import Image
import cv2
import numpy as np
import time
import os

start = time.time()


# In[2]:


PAI_LIST = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
             '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
             '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
             '1z', '2z', '3z', '4z', '5z', '6z', '7z', '0m', '0p', '0s']


# 

# In[ ]:


screenshot = pyautogui.screenshot('screenshot.png')
img = Image.open('screenshot.png')

# for index in range(6):
#     x1 = first_pai_x - (pai_width * index) - (pai_diff * index)
#     x2 = first_pai_x + pai_width - (pai_width * index) - (pai_diff * index)
#     im_crop = img.crop((x1, y1, x2, y2))
#     im_crop.save('pais/' + str(index) + '.png', quality=95)


# In[ ]:



    


# In[ ]:


def grayscale_matching(index, temp, img):

    img_copy = img.copy()
    
#     切り抜きを保存
    cv2.imwrite("pais/" + str(index) + ".png", temp)
        
#     グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # テンプレート画像の高さ・幅
    h, w = temp_gray.shape
    
    # テンプレートマッチング（OpenCVで実装）
    match = cv2.matchTemplate(gray, temp_gray, cv2.TM_SQDIFF)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = min_pt
    
    # # テンプレートマッチングの結果を出力
    cv2.rectangle(img_copy, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv2.imwrite("pais/result" + str(index) + ".png", img_copy)


# - 1牌目→横1600,1670、縦563,621
# - 二段目→横1600,1665、縦490,551

# In[ ]:


img_path = "pais/pai_list.jpg"
# img_path = "screenshot.png"
if os.path.isfile(img_path):
    img = cv2.imread(img_path)
else:
    print(img_path, 'ファイルが存在しません')
    
# スクショの読み込み
shot = cv2.imread('screenshot.png')

# screenshot = pyautogui.screenshot()
# shot = pil2cv(screenshot)

# display_pais = []



# 一段目
y1 = 563
y2 = 623
# pai_height = 60
pai_width = 71
first_pai_x = 1600
pai_diff = 9

for index in range(6):
    #     スクショからの切り抜き
    x1 = first_pai_x - (pai_width * index) - (pai_diff * index)
    x2 = first_pai_x + pai_width - (pai_width * index) - (pai_diff * index)
    temp = shot[y1:y2, x1:x2, :]
#     実行
    pai = grayscale_matching(index, temp, img)
    
# 2段目
y1 = 495
y2 = 551
# pai_height = 60
pai_width = 65
first_pai_x = 1600
pai_diff = 11

for index in range(6):
    #     スクショからの切り抜き
    x1 = first_pai_x - (pai_width * index) - (pai_diff * index)
    x2 = first_pai_x + pai_width - (pai_width * index) - (pai_diff * index)
    temp = shot[y1:y2, x1:x2, :]
#     実行
    pai = grayscale_matching(index + 6, temp, img)

# 3段目
y1 = 432
y2 = 484
# pai_height = 60
pai_width = 65
first_pai_x = 1600
pai_diff = 11

for index in range(5):
    #     スクショからの切り抜き
    x1 = first_pai_x - (pai_width * index) - (pai_diff * index)
    x2 = first_pai_x + pai_width - (pai_width * index) - (pai_diff * index)
    temp = shot[y1:y2, x1:x2, :]
#     実行
    pai = grayscale_matching(index + 12, temp, img)

