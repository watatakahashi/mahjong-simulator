#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyautogui
from PIL import Image
import cv2
import numpy as np
import time
import os
from requests_html import HTMLSession


start = time.time()


# In[2]:


PAI_LIST = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
             '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
             '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
             '1z', '2z', '3z', '4z', '5z', '6z', '7z', '0m', '0p', '0s', 'x']
PAI_LIST_X = [1,140,278,415,558,696,833,967,1104,
             1241,1380,1518,1657,1796,1934,2072,2208,2347,
             2487,2625,2767,2905,3042,3181,3319,3458,3595,
             3732,3869,4004,4151,4291,4427,4570,4705,4840,4990, 5130]


# In[ ]:


# screenshot = pyautogui.screenshot('screenshot.png')
# print('screenshot=', screenshot)

# 指定範囲を取る場合
# sc = pyautogui.screenshot(region=(100, 200, 300, 400))


# In[ ]:


# img = Image.open('screenshot.png')

# 横幅
y1 = 1590
y2 = 1787
pai_width = 135
first_pai_x = 350
pai_diff = 5

tsumo_x = 2220

# for index in range(13):
#     x1 = first_pai_x + (pai_width * index) + (pai_diff * index)
#     x2 = first_pai_x + pai_width * (index + 1) + (pai_diff * index)
#     im_crop = img.crop((x1, y1, x2, y2))
#     im_crop.save('pais/' + str(index) + '.png', quality=95)


# In[ ]:


def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return idx
#     return list[idx]


# In[ ]:


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


# In[ ]:


def save_pai_image(img):
    ''' 切り取った牌をを保存する '''
    pass


# In[ ]:


def pai_efficiency(pais):
    
    url = "https://tenhou.net/2/?q=" + str(pais)
    # セッション開始
    session = HTMLSession()
    r = session.get(url)
    r.html.render()

    textarea_elem = r.html.find("textarea", first=True)
    print(textarea_elem.html)   


# In[ ]:


def remove_x_pai(pais_str):
    """ ツモってないxを削除する """
    return pais_str.replace('x', '')


# # テンプレートマッチング実行

# In[4]:


def grayscale_matching(index, temp, img):
    img_copy = img.copy()
#     切り抜きの保存
#     cv2.imwrite("pais/" + str(index) + ".png", temp)

#     赤判定用
#     img_bgr = cv2.split(temp)
#     print(index, '赤色平均', np.mean(img_bgr[2]))
    
#     hantei = img_bgr[2] - (img_bgr[0] + img_bgr[1]) / 2
#     print(index, '判定値', np.count_nonzero(hantei > 0))
    
#     グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp_gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # テンプレート画像の高さ・幅
    h, w = temp_gray.shape
    
    # テンプレートマッチング（OpenCVで実装）
    match = cv2.matchTemplate(gray, temp_gray, cv2.TM_SQDIFF)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = min_pt
    
    pai = PAI_LIST[getNearestValue(PAI_LIST_X, pt[0])]
    
    # # テンプレートマッチングの結果を出力
#     cv2.rectangle(img_copy, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
#     cv2.imwrite("pais/result" + str(index) + ".png", img_copy)
    return pai


# In[ ]:


def color_matching(index):
    temp = cv2.imread('pais/' + str(index) + '.png')
    img2 = img.copy()
    
    #カラー読み込みなのでひとつ要素が多い。
    s,w,h=temp.shape[::-1]

    res = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED)
    #max_locが画像の左上からの位置。
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    pt = max_loc
    print(index, '.pngの座標イチ', pt)
    

    #黒色(0,0,0)の四角で囲む。
    cv2.rectangle(img2, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,0,200),4)
    cv2.imwrite("pais/result" + str(index) + ".png", img2)
   


# In[ ]:


def main(img):
    # スクショの読み込み
    # shot = cv2.imread('screenshot.png')

    screenshot = pyautogui.screenshot()
    # screenshot = pyautogui.screenshot('screenshot.png')
    shot = pil2cv(screenshot)
    display_pais = []
    
    for index in range(14):
        #     スクショからの切り抜き
        x1 = first_pai_x + (pai_width * index) + (pai_diff * index)
        x2 = first_pai_x + pai_width * (index + 1) + (pai_diff * index)
#         ツモ牌のみ例外
        if index == 13:
            x1, x2 = tsumo_x, tsumo_x + pai_width
        temp = shot[y1:y2, x1:x2, :]
        
#         実行
        pai = grayscale_matching(index, temp, img)
        display_pais.append(pai)
#     print(display_pais)
    
#     ツモ牌がある場合は牌効率にかける
    try:
        x_index = display_pais.index('x')
        display_pais = display_pais[:x_index]
    except:pass
#     print(display_pais)
#     print('手牌数=', len(display_pais))
    if len(display_pais) in [5, 8, 11, 14]:
        print(display_pais)
#         print('牌効率算出')
        display_pais_str = ''.join(display_pais)
        pai_efficiency(display_pais_str)


# # メイン処理

# In[ ]:


print('実行開始')

img_path = "pais/pai_list.jpg"
if os.path.isfile(img_path):
    img = cv2.imread(img_path)
else:
    print(img_path, 'ファイルが存在しません')

for i in range(100):
#     print(i,'回目')
    main(img)
print('実行終了')


# In[ ]:


# # グレースケール変換
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# # テンプレート画像の高さ・幅
# h, w = temp.shape

# # テンプレートマッチング（OpenCVで実装）
# match = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF)
# min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
# pt = min_pt

# # テンプレートマッチングの結果を出力
# cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
# cv2.imwrite("pais/result.png", img)


# In[ ]:


elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

