import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first = True)

        # the linear layer that maps the hidden state output dimension 
        # to the number of words we want as output, vocab size as need to predict all
        self.hidden2word = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        
        # In the code cell below, outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]. 
        # Your output should be designed such that outputs[i,j,k] contains the model's predicted score, 
        # indicating how likely the j-th token in the i-th caption in the batch is the k-th token in the vocabulary. 
        # the lstm takes in our embeddings and hiddent state
        
        #embed the captions
        embeds = self.word_embeddings(captions)
        
        embeds_remove_front = embeds[:, :-1, :]
        inputs = torch.cat((features.unsqueeze(1), embeds_remove_front), dim=1)
        lstm_out, hidden_state = self.lstm(inputs)
        return self.hidden2word(lstm_out)
        
            
    def sample(self, inputs, states=None, max_len=20):
        words = []
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        for counter in range(max_len):
            if counter == 0:
                lstm_out, hidden_state = self.lstm(inputs, states)
            else:
                lstm_out, hidden_state = self.lstm(embeds, hidden_state)
            word_scores = self.hidden2word(lstm_out)
            prob, word = word_scores.max(2)
            embeds = self.word_embeddings(word)
            words.append(word.item())
            if word.item() == 1:
                break
            
        return words
