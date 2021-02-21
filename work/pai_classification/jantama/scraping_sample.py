#!/usr/bin/env python
# coding: utf-8

# # スクレイピングのサンプル

# In[ ]:


from requests_html import HTMLSession

url = "https://tenhou.net/2/?q=2335m4568p1258s6z6p"

# セッション開始
session = HTMLSession()
r = session.get(url)
r.html.render()

textarea_elem = r.html.find("textarea", first=True)
print(textarea_elem.html)   


# In[ ]:




