#- *- coding: utf-8 -*-
import numpy as np
import logging
from tqdm import tqdm
from model.utils.utils import *
from model.bilstm import *
import torch.optim as optim
import pickle

EMBEDDING_DIM = 300
HIDDEN_DIM = 300
batch_size = 32

isSplit = True

lang = Lang()
train, dev = prepare_data_seq("data/train/train.sgml", lang,  False, batch_size)

model = BiLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, lang.n_words, lang.n_tags)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_function = nn.NLLLoss(weight=torch.Tensor([1,5]).cuda())
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

with open("modeldict.pkl","wb") as file:
     pickle.dump(lang, file, pickle.HIGHEST_PROTOCOL)

global_f1 = 0.0
num_epoch = 0
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    num_epoch += 1
    print("The {} epoch in Training".format(epoch))
    batch_num = 0
    model.train()
    for  src_index, src_length, target_index, target_length, _, _ in train:
        batch_num += 1
        model.zero_grad()
        tag_scores = model(src_index, src_length)
        target_scores = target_index.transpose(0, 1)

        total_loss = 0.0
        for i in range(len(src_length)):
            total_loss += loss_function(tag_scores[i][:src_length[i]], target_scores[i][:src_length[i]])

        loss = total_loss / len(src_length)
        loss.backward()

        if batch_num % 10000 == 0:
            print("The {} batch is {}".format(batch_num, loss.data.item()))
        optimizer.step()

    if num_epoch % 5 != 0:
        continue

    model.eval()
    print("============= Begin Validation =========================\n")
    if True:
        # See what the score are after training
        with torch.no_grad():
            correct_num = 0
            total_num = 0
            prediction_num = 0
            for batch_data in dev:
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
            if f1 > global_f1:
                save_model(model, str(f1)+"-"+str(precision)+"-"+str(recall))
                print("Recall is {}".format(str(recall)))
                print("Precision is {}".format(str(precision)))
                print("F1 is {}".format(str(f1)))
                glabal_f1 = f1
