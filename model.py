import torch
import torch.nn as nn
import numpy as np


class CNN_ReLU_BatchNorm(nn.Module):
    def __init__(self):
        super(CNN, self).__init__(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.cnn = nn.Sequential(
                      nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                      nn.ReLU(),
                      nn.BatchNorm1d(out_channels),
                   )

    def forward(self, feature):
        feature = torch.permute(feature, (0, 2, 1))
        feature = self.cnn(feature)
        feature = torch.permute(feature, (0, 2, 1))
        return feature
        
class BidirectionalLSTM(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, num_layers=2):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent
        
class TS_VAD(nn.Module):
    def __init__(self, rproj=128, nproj=32, cell=896):
        super(TS_VAD, self).__init__()
        self.cnn1 = CNN_ReLU_BatchNorm(in_channels=40, out_channels=64)
        self.cnn2 = CNN_ReLU_BatchNorm(in_channels=64, out_channels=64)
        self.cnn3 = CNN_ReLU_BatchNorm(in_channels=64, out_channels=128)
        self.cnn4 = CNN_ReLU_BatchNorm(in_channels=128, out_channels=128)

        self.pooling = nn.MaxPool1d(kernel_size=2)

        self.linear = n.Linear(228, 3*rproj)
        self.rnn_speaker_detection = BidirectionalLSTM(3*rproj, cell)

        self.rnn_combine = BidirectionalLSTM(8*cell, cell)

        self.cnn = nn.Sequential(
                      self.cnn1,
                      self.cnn2,
                      self.cnn3,
                      self.pooling,
                      self.cnn4
                   )

    def forward(self, ivectors, feature):
        feature = self.cnn(feature)
        combine = torch.tensor([])
        for ivector in ivectors:
            x = torch.cat(feature, ivector)
            combine = torch.cat(x, dim=-1)


            
if __name__ == '__main__':
    feature = torch.randn((32, 100, 40)) # B x T x D
    ivectors = torch.randn((4, 32, 100)) # S x B x D
    model = TS_VAD()
    output = model(ivectors, feature)
