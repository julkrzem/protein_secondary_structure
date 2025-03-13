The aim of the project was to create a simple app that will return predicted proteins structures based on the sequences. 
At first started with some tests to see how different model architectures perform and I checked different variants to finally choose one that was having the highest character-level accuracy. 
The process is documented in notebooks in the Notebooks folder.

## Dataset

The dataset used for model training was downloaded from Kaggle.com: https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure/
I used the cleaned data to simplify the process on the initial stage of the project. 
The dataset consists of over 300000 samples of polypeptide chains. For the model development I chose the sequences not longer than 100 amino acids, for the X varaible I used only amino acid sequences and for y - three-state (Q3) secondary structure. 
With more computational power it would probably be even better to use sst8 and longer sequences or even the whole dataset. 
I deduplicated the sequences in order to prevent data leakage, as a result removing multiple identical chains coming from the same proteins, and potentially also some short identical sequences.

## Model

I experimented a bit with encoder-decoder architecture interpreting this task as seq2seq problem, however the results were not as impressive because the I-O pairs do not have variable length as in typical NLP translation problem. So that property of seq2seq is actually not beneficial in this case, and training effective network that way would probably require way more resources than necessary. 

The encoder-decoder model was based on PyTorch seq2seq tutorial (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), with some modifications in the structure as well as the training process. I used bidirectional network as I expected better results, tried different dimensions and hyperparameters. Hovewer the task itself was more similar to sequence labeling, with both inputs having the same length, because of that I decided to try simpler architecture with only one recurrent unit.

In later experiments tried LSTM model. I experimented with hyperparameters and different training processes - normal training and also using dynamic padding (to maximal length in the batch), with the aim of minimising the effect of excessive padding with short sequences. The final results were slightly lower than with the normal training so I saved and used the simple LSTM as the final model for the project. 






*The project was developed for educational purpose only.*
