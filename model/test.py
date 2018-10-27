#- *- coding: utf-8 -*-
import numpy as np
import logging
from tqdm import tqdm
from model.utils.utils import *
from model.bilstm import *
import pickle

EMBEDDING_DIM = 300
HIDDEN_DIM = 300
batch_size = 32

saved_model_path = 'save/model.th'
saved_model_dict = "save/modeldict"

with open("modeldict.pkl", "rb") as file:
    lang = pickle.load(file)

test13 = prepare_data_seq("data/test/test13.sgml", lang, True, batch_size)
test14 = prepare_data_seq("data/test/test14.sgml", lang, True, batch_size)
test15 = prepare_data_seq("data/test/test15.sgml", lang, True, batch_size)

model = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, lang.n_words, lang.n_tags)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if USE_CUDA:
    logging.info("MODEL {} LOADED".format(saved_model_path))
    model = torch.load(saved_model_path)
else:
    logging.info("MODEL {} LOADED".format(saved_model_path))
    model =  torch.load(saved_model_path,lambda storage, loc: storage)

model.eval()
with torch.no_grad():
    correct_num = 0
    total_num = 0
    prediction_num = 0
    for batch_data in test13:
        model.zero_grad()
        model.hidden = model.get_state(batch_data[0])
        prediction_scores = model(batch_data[0], batch_data[1])
        prediction_scores = torch.argmax(prediction_scores, dim=2)
        target_scores = batch_data[2].transpose(0, 1)
        for idx, sen_len in enumerate(batch_data[3]):
            for i in range(sen_len):
                if target_scores[idx][i].item() == 1:
                    total_num += 1
                    if prediction_scores[idx][i] == target_scores[idx][i]:
                        correct_num += 1
                if prediction_scores[idx][i].item() == 1:
                    prediction_num += 1

    precision = 1.0 * correct_num / prediction_num
    recall = 1.0 * correct_num / total_num
    f1 = 2 * recall * precision / (recall + precision)

    print("==========Test13 Result ==============\n")
    print("Recall is {}\n".format(str(recall)))
    print("Precision is {}\n".format(str(precision)))
    print("F1 is {}\n".format(str(f1)))

with torch.no_grad():
    correct_num = 0
    total_num = 0
    prediction_num = 0
    for batch_data in test14:
        model.zero_grad()
        model.hidden = model.get_state(batch_data[0])
        prediction_scores = model(batch_data[0], batch_data[1])
        prediction_scores = torch.argmax(prediction_scores, dim=2)
        target_scores = batch_data[2].transpose(0, 1)
        for idx, sen_len in enumerate(batch_data[3]):
            for i in range(sen_len):
                if target_scores[idx][i].item() == 1:
                    total_num += 1
                    if prediction_scores[idx][i] == target_scores[idx][i]:
                        correct_num += 1
                if prediction_scores[idx][i].item() == 1:
                    prediction_num += 1

    precision = 1.0 * correct_num / prediction_num
    recall = 1.0 * correct_num / total_num
    f1 = 2 * recall * precision / (recall + precision)

    print("==========Test14 Result ==============\n")
    print("Recall is {}\n".format(str(recall)))
    print("Precision is {}\n".format(str(precision)))
    print("F1 is {}\n".format(str(f1)))

with torch.no_grad():
    correct_num = 0
    total_num = 0
    prediction_num = 0
    for batch_data in test15:
        model.zero_grad()
        model.hidden = model.get_state(batch_data[0])
        prediction_scores = model(batch_data[0], batch_data[1])
        prediction_scores = torch.argmax(prediction_scores, dim=2)
        target_scores = batch_data[2].transpose(0, 1)
        for idx, sen_len in enumerate(batch_data[3]):
            for i in range(sen_len):
                if target_scores[idx][i].item() == 1:
                    total_num += 1
                    if prediction_scores[idx][i] == target_scores[idx][i]:
                        correct_num += 1
                if prediction_scores[idx][i].item() == 1:
                    prediction_num += 1

    precision = 1.0 * correct_num / prediction_num
    recall = 1.0 * correct_num / total_num
    f1 = 2 * recall * precision / (recall + precision)

    print("==========Test15 Result ==============\n")
    print("Recall is {}\n".format(str(recall)))
    print("Precision is {}\n".format(str(precision)))
    print("F1 is {}\n".format(str(f1)))

