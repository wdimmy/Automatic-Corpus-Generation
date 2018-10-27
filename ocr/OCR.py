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
from strTools import *
import random
import argparse
from tqdm import tqdm

from handler import * 


parse = argparse.ArgumentParser(description="Please input the path where the original data locates and the path where the generated file will be saved!") 
parser.add_argument('-i','--input', help='the original data path', required=True)
parser.add_argument('-o','--output', help='the save file path', required=True)

args = vars(parser.parse_args())

file_path = args["input"]
save_path = args["output"]


# this array is used for deciding the number  of errors for a given sentence
error_nums = [1,2,3,4]
stopwords = getPunctuation()
spliter = ["。","？","！"]
ocrFile = codecs.open(save_path,"w","utf-8")
total_error_num = 0
total_line_num = 0


with tqdm(os.path.getsize(file_path)) as pbar:
    with codecs.open(file_path,"r","utf-8") as file:
        for passage in file:
            total_line_num += 1
            pbar.update(len(passage))
            passage = "".join(passage.strip().split())
            # In this part, we split the passage into segments using the punctuations as spliters
            segmentments = []
            tmp_seg = ""
            for idx in range(len(passage)):
                if passage[idx] in spliter:
                    tmp_seg += passage[idx]
                    if len(tmp_seg) >= 8 and len(tmp_seg) <= 85:
                          segmentments.append(tmp_seg)
                          tmp_seg = ""
                else:
                    tmp_seg += passage[idx]

            if len(tmp_seg) >= 8 and len(tmp_seg) <= 85:
                segmentments.append(tmp_seg)
            for line in segmentments:
                result = set()

                line_len = len(line)
                error_num = random.choice(error_nums)
                used_index = []
                index_list = [i for i in range(line_len)]

                # This variable is used for controlling the number of iterations in the while loop. If the number of while loops is over the specified number, we will break the loop
                turn = 0

                # This variable is used for count the number of incorrectly-detected characters. If the number reaches the value of "error_num", we will break the loop
                hits = 0
                while True:
                    turn += 1
                    i = random.choice(index_list) # randomize an index for detection
                    if i not in used_index and turn < 10 and hits < error_num:
                        str_to_detect = line[i:i+4]
                        if len(str_to_detect) < 2 or not is_chinese(str_to_detect):
                            continue
                        else:
                            characterToimage(str_to_detect, "tmp")
                            blur("tmp")
                            detected_result = pytesseract.image_to_string(Image.open('blurred.png'), lang='chi_sim')
                            if len(str_to_detect) == len(detected_result):
                                detected_error = 0
                                bias = 0
                                for j in range(len(str_to_detect)):
                                    if detected_result[j] != str_to_detect[j]:
                                        detected_error += 1
                                        bias = j
                                if detected_error == 1 and str_to_detect[bias] != "一" and detected_result[bias] != "_" and str_to_detect[bias] not in stopwords:
                                    result.add((i+bias, str_to_detect[bias], detected_result[bias]))
                                    error_num += 1
                                    used_index.append(i+bias)

                        used_index.append(i)
                        turn += 1
                    if turn >= 10:
                        break

                if len(result) > 0:
                    total_error_num += 1
                    ocrFile.write(line.strip() + "\n")
                    new_result = [str(item[0])+":"+item[1] +","+ item[2] for item in result]
                    ocrFile.write(";".join(new_result)+"\n")
ocrFile.close()

