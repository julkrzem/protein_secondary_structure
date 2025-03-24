import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super().__init__()

        self.inpit_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, output_size)

    def forward(self, x):
        x = self.embed(x)
        x = F.relu(x)
        x , _ = self.lstm(x)
        x = self.fc(x)

        x = F.log_softmax(x, dim = -1)
        return x
    

def indexFromSequence(lang, sequence):
    encoded = []
    for char in list(sequence):
        try:
            encoded.append(lang[char])
        except KeyError:
            encoded.append(lang["*"])
    return encoded

def charFromIndex(lang,indexes):
    return [lang[str(int(char))] for char in indexes]


def preprocess_data(input_lang, input_data):
    input_data = indexFromSequence(input_lang,input_data.upper())
    padding_array = np.zeros((1, 101), dtype=np.int32) # for single input
    for idx, input in enumerate(input_data):
        padding_array[0,idx] = input
    x = torch.tensor(padding_array, dtype=torch.long)
    return x

def return_predictions(output_lang, output_data):
    if output_data.dim() !=0:
            out = output_data[output_data.nonzero()].squeeze()
            if out.nelement() == 0:
                return "predict result is empty, try longer sequence"
            output_char = charFromIndex(output_lang,out)
    else:
        output_char = output_lang[str(int(output_data))]
    output = "".join(output_char)
    return output

def model_predict(input_data, input_lang, output_lang):
    with torch.no_grad():
        x = preprocess_data(input_lang, input_data)
        output_pred = model(x)
        _, topi = output_pred.topk(1)
        result=topi.squeeze()
        y = return_predictions(output_lang, result)
    return y


model = LSTM(22, 64, 256, 4)
model.load_state_dict(torch.load("Models/lstm_1.pth"))
model.eval()

