import nltk
from nltk.stem.porter import PorterStemmer
import json
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bot_function import *
from domain_function import *

FILE = "data1.pth"
data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
model_state = data["model_state"]
tags = data["tags"]
model = Neturalnet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def chatbot_response(sentence1):
    msg = ''
    sentence = tokenize(sentence1)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                msg = random.choice(intent['responses'])
    else:
        msg = "Your question is out of scope or check your spelling"
    return msg
