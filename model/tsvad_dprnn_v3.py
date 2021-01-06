import sys
import torch
import numpy as np
import torch.nn as nn


class CNN_ReLU_BatchNorm(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, stride=(1, 1), padding=1):
        super(CNN_ReLU_BatchNorm, self).__init__()
        self.cnn = nn.Sequential(
                      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                      nn.ReLU(),
                      nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
                   )

    def forward(self, feature):
        feature = self.cnn(feature)
        return feature
     

class BLSTMP(nn.Module):
    def __init__(self, n_in, n_hidden, nproj=384, dropout=0, num_layers=1):
        super(BLSTMP, self).__init__()

        self.num_layers = num_layers

        self.rnns = nn.ModuleList([nn.LSTM(n_in, n_hidden, dropout=dropout, bidirectional=True, batch_first=True)])
        self.linears = nn.ModuleList([nn.Linear(n_hidden*2, nproj)])

        for i in range(num_layers-1):
            self.rnns.append(nn.LSTM(nproj, n_hidden, dropout=dropout, bidirectional=True, batch_first=True))
            self.linears.append(nn.Linear(n_hidden*2, nproj))
    
    def forward(self, feature):
        recurrent, _ = self.rnns[0](feature)
        output = self.linears[0](recurrent)

        for i in range(self.num_layers-1):
            output, _ = self.rnns[i+1](output)
            output = self.linears[i+1](output)
        
        return output


class DPRNN(nn.Module):
    def __init__(self, nproj=384, cell=896, num_layers=6):
        super(DPRNN, self).__init__()

        self.num_layers = num_layers

        self.dprnn_t = nn.ModuleList([BLSTMP(nproj, cell) for i in range(num_layers)])
        self.t_norm = nn.ModuleList([nn.GroupNorm(1, nproj, eps=1e-8) for i in range(num_layers)])
        self.dprnn_c = nn.ModuleList([BLSTMP(nproj, cell) for i in range(num_layers)])
        self.c_norm = nn.ModuleList([nn.GroupNorm(1, nproj, eps=1e-8) for i in range(num_layers)])

    def forward(self, output):
        bs, dim, tframe, spks = output.size()
        for i in range(self.num_layers):
            dprnn_t_in = output.permute(0, 3, 2, 1).contiguous().view(bs*4, tframe, -1)    # B*4 x T x 384
            dprnn_t_out = self.dprnn_t[i](dprnn_t_in).contiguous().view(bs, 4, tframe, -1) # B x 4 x T x 384
            dprnn_t_out = dprnn_t_out.permute(0, 3, 2, 1)                                  # B x 384 x T x 4
            
            dprnn_t_out = self.t_norm[i](dprnn_t_out)
            output = dprnn_t_out + output                                                  # B x 384 x T x 4
            
            dprnn_c_in = output.permute(0, 2, 3, 1).contiguous().view(bs*tframe, 4, -1)    # B*T x 4 x 384
            dprnn_c_out = self.dprnn_c[i](dprnn_c_in).contiguous().view(bs, tframe, 4, -1) # B x T x 4 x 384
            dprnn_c_out = dprnn_c_out.permute(0, 3, 1, 2)                                  # B x 384 x T x 4

            dprnn_c_out = self.c_norm[i](dprnn_c_out)
            output = dprnn_c_out + output                                                  # B x 384 x T x 4

        output = output.permute(0, 2, 3, 1) # B x T x 4 x 384
        
        return output
        

class Model(nn.Module):
    def __init__(self, out_channels=[64, 64, 128, 128], nproj=384, cell=896, dprnn_layers=6):
        super(Model, self).__init__()

        batchnorm = nn.BatchNorm2d(1, eps=0.001, momentum=0.99)

        cnn_relu_batchnorm1 = CNN_ReLU_BatchNorm(in_channels=1, out_channels=out_channels[0])
        cnn_relu_batchnorm2 = CNN_ReLU_BatchNorm(in_channels=out_channels[0], out_channels=out_channels[1])
        cnn_relu_batchnorm3 = CNN_ReLU_BatchNorm(in_channels=out_channels[1], out_channels=out_channels[2], stride=(1, 2))
        cnn_relu_batchnorm4 = CNN_ReLU_BatchNorm(in_channels=out_channels[2], out_channels=out_channels[3])
        
        self.cnn = nn.Sequential(
                      batchnorm,
                      cnn_relu_batchnorm1,
                      cnn_relu_batchnorm2,
                      cnn_relu_batchnorm3,
                      cnn_relu_batchnorm4
                   )
        
        self.linear = nn.Linear(out_channels[-1]*20+100, nproj)
        self.rnn_speaker_detection = BLSTMP(nproj, cell, num_layers=2)
        self.rnn_combine = DPRNN(num_layers=dprnn_layers)
        self.output_layer = nn.Linear(nproj, 1)

    def forward(self, batch):
        feats, targets, ivectors = batch

        feats = self.cnn(feats)
        bs, chan, tframe, dim = feats.size()
        
        feats    = feats.permute(0, 2, 1, 3)
        feats    = feats.contiguous().view(bs, tframe, chan*dim) # B x 1 x T x 2560
        feats    = feats.unsqueeze(1).repeat(1, 4, 1, 1)         # B x 4 x T x 2560
        ivectors = ivectors.view(bs, 4, 100).unsqueeze(2)        # B x 4 x 1 x 100
        ivectors = ivectors.repeat(1, 1, tframe, 1)              # B x 4 x T x 100
        
        sd_in  = torch.cat((feats, ivectors), dim=-1)            # B x 4 x T x 2660
        sd_in  = self.linear(sd_in).view(4*bs, tframe, -1)       # B*4 x T x 384
        sd_out = self.rnn_speaker_detection(sd_in)               # B*4 x T x 384
        
        # DPRNN
        output = sd_out.contiguous().view(bs, 4, tframe, -1)     # B x 4 x T x 384
        output = output.permute(0, 3, 2, 1)                      # B x 384 x T x 4
        output = self.rnn_combine(output)                        # B x T x 4 x 384 

        preds   = self.output_layer(output).squeeze(-1)          # B x T x 4
        preds   = nn.Sigmoid()(preds)
        
        loss = nn.BCELoss(reduction='sum')(preds, targets) / tframe / bs
        loss_detail = {"diarization loss": loss.item()}
        
        return loss, loss_detail

    def inference(self, batch): 
        _, feats, ivectors = batch

        feats = self.cnn(feats)
        bs, chan, tframe, dim = feats.size()
        
        feats    = feats.permute(0, 2, 1, 3)
        feats    = feats.contiguous().view(bs, tframe, chan*dim) # B x 1 x T x 2560
        feats    = feats.unsqueeze(1).repeat(1, 4, 1, 1)         # B x 4 x T x 2560
        ivectors = ivectors.view(bs, 4, 100).unsqueeze(2)        # B x 4 x 1 x 100
        ivectors = ivectors.repeat(1, 1, tframe, 1)              # B x 4 x T x 100
        
        sd_in  = torch.cat((feats, ivectors), dim=-1)            # B x 4 x T x 2660
        sd_in  = self.linear(sd_in).view(4*bs, tframe, -1)       # B*4 x T x 384
        sd_out = self.rnn_speaker_detection(sd_in)               # B*4 x T x 384
        
        # DPRNN
        output = sd_out.contiguous().view(bs, 4, tframe, -1)     # B x 4 x T x 384
        output = output.permute(0, 3, 2, 1)                      # B x 384 x T x 4
        output = self.rnn_combine(output)                        # B x T x 4 x 384 

        preds   = self.output_layer(output).squeeze(-1)          # B x T x 4
        preds   = nn.Sigmoid()(preds)
        
        return preds
