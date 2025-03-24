import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from io import BytesIO

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
    n = len(input_data)
    padding_array = np.zeros((n, 101), dtype=np.int32)
    for index, seq in enumerate(input_data):
        seq = indexFromSequence(input_lang,seq.upper()[:100])
        m = len(seq)
        padding_array[index,0:m] = np.array(seq)
    x = torch.tensor(padding_array, dtype=torch.long)
    return x


def return_predictions(output_lang, output_data):
    pred = []
    for seq in output_data:
        if seq.dim()!=0:
            out = seq[seq.nonzero()].squeeze()
            output_char = charFromIndex(output_lang,out)
        else:
            output_char = output_lang[str(int(seq))]
        output = "".join(output_char)
        pred.append(output)
    return pred


def model_predict(input_data, input_lang, output_lan, model):
    with torch.no_grad():
        x = preprocess_data(input_lang, input_data)
        outputs_pred = model(x)
        _, topi = outputs_pred.topk(1)
        if len(input_data) == 1:
            result = topi.squeeze(-1)
        else: result = topi.squeeze()
        y = return_predictions(output_lan, result)
    return y

def read_fasta(file):
    fasta = file.read().decode('utf-8')
    seq = str()
    ids = []
    for line in fasta.split("\n"):
        if ">" not in line:
            seq += line.strip("\n")
        else:
            seq += " "
            ids.append(line.strip("\n"))
    x = seq.split(" ")[1:]
    for idx, sq in enumerate(x):
        x[idx] = sq[:100]
    return ids, x

def print_results(color_map, sequence, structure):
    
    x = np.arange(len(sequence))

    fig, ax = plt.subplots(figsize=(14, 2))

    for i, char in enumerate(structure):
        ax.bar(x[i], 1, width=1, bottom=0, color=color_map[char])

    ax.set_xticks(x)
    ax.set_xticklabels(list(structure))
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(structure) - 0.5)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    ax2 = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(sequence), verticalalignment='bottom')
    plt.subplots_adjust(top=0.85, bottom=0.15)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img