# -*- coding: utf-8 -*-
import os 
import pygame
import cv2
import codecs
import time
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
from PIL import Image, ImageFilter

def getLocation(x):
    if x == 1:
        #mask the upper part
        left_x = 0
        left_y = 0
        right_x = 200
        right_y = 70
        return left_x,left_y, right_x, right_y
    elif x == 2:
        #mask the bottom part
        left_x = 0
        left_y = 120
        right_x = 200
        right_y = 265
        return left_x, left_y, right_x, right_y
    elif x == 3:
        #mask the left part
        left_x = 0
        left_y = 0
        right_x = 100
        right_y = 265
        return left_x, left_y, right_x, right_y
    elif x == 4:
        #mask the central part
        left_x = 0
        left_y = 80
        right_x = 200
        right_y = 140
        return left_x, left_y, right_x, right_y
    else:
        # mask the right part
        left_x = 100
        left_y = 0
        right_x = 200
        right_y = 265
        return left_x, left_y, right_x, right_y

def blur(imagename):
    image = Image.open(imagename+".png")
    num = random.randint(1,5)
    left_x, left_y,right_x,right_y = getLocation(num)

    left_x += 0
    right_x += 0
    crop_image = image.crop((left_x,left_y, right_x, right_y))
    blured_image = crop_image.filter(ImageFilter.GaussianBlur(radius=10))
    image.paste(blured_image, (left_x,left_y, right_x, right_y))
    image.save("blurred.png")


pygame.init()
def characterToimage(word,imagename):
	font = pygame.font.Font("msyh.ttc", 200) # you can load HYKaiTiJ.ttf to generated Kai Font Charaters
	rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
	pygame.image.save(rtext, imagename+".png")


def getPunctuation():
	stopwords = ["阿","啊","呃","欸","哇","呀","也","耶","哟","欤","呕","噢","呦","嘢","吧","罢","呗","啵","的","家","啦","来","唻","了","嘞","哩","咧","咯","啰","喽","吗","嘛","嚜","么","哪","呢","呐","否","呵","哈","不","兮","般","则","连","罗","噻","哉","呸"]
	# punc = ['？','。','。。。','！','’','，','…','《','》', '‘',
	# 		'；', '，', '“', '”', '）', '（', '·', '~', '％',
	# 		'：', '．', '—', '、', '～', '°', '　']
    #
	# for c in string.punctuation:
	# 	punc.append(c)
    #
	# stopwords.extend(punc)
	# stopwords = set(stopwords)

	return stopwords