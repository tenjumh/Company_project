#!/usr/bin/env python
# coding: utf-8

# In[17]:


# import the necessary packages
import numpy as np
import urllib
import cv2
from flickrapi import FlickrAPI
import ssl, os, sys
from pprint import pprint

FLICKER_KEY = '505e8c2b05e8b5aa51766a43fc4bc7a7'
FLICKER_SECRET = '46b3ef2f43b04ac6'

flickr = FlickrAPI(FLICKER_KEY, FLICKER_SECRET, format='parsed-json')
extras='url_s'
# extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
# download the image, convert it to a NumPy array, and then read
# it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

args = sys.argv[1:]
for keyword in args:
    def imageCrawling(keyword):
        page = 100
        path = './data/train/'
        os.mkdir(path + keyword)
        target = flickr.photos.search(text=keyword, per_page=page, extras=extras)
        photos = target['photos']
        
        for i in range(page):
            image_original_url = photos['photo'][i]['url_s']
            image_temp = url_to_image(image_original_url)
            file_name = './data/train/{}/{}_{}.jpg'.format(keyword, keyword, i)
            cv2.imwrite(file_name, image_temp)
            # cv2.imshow('image_temp', image_temp)
            # cv2.waitKey(0)
    imageCrawling(keyword)

# pprint(photos)


# In[ ]:


'''
# initialize the list of image URLs to download
urls = [
    "https://live.staticflickr.com/65535/48239557012_4408050887_m.jpg",
    "https://live.staticflickr.com/65535/48239467791_c85039d01f_m.jpg"
]

# loop over the image URLs
for url in urls:
    print('download:', url)
    image = url_to_image(url)
    cv2.imshow('image', image)
    cv2.waitKey(0)
'''


# In[ ]:




