# -*- coding: utf-8 -*-
"""
author: Lijie Wang

function: brainprint recognition

date: 2023.04.25

condition: 9channels & 3s
"""

import sys
sys.path.append('..')
import torch
import datetime
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data.dataset as dataset

 
# define data class
class subDataset(dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label
    

# define model
class rnn_classify(nn.Module):
     def __init__(self, in_feature=9, hidden_feature=100, num_class=35, num_layers=2):
          super(rnn_classify, self).__init__()
          self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)
          self.classifier = nn.Linear(hidden_feature, num_class)
          
     def forward(self, x):
          x = x.permute(1, 0, 2)
          out, _ = self.rnn(x)
          out = out[-1,:,:]
          out2 = self.classifier(out)
          return out2

 
# calculate accuracy
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
    
# define train process
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            single_sub_acc = []
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda())
                    label = Variable(label.cuda())
                else:
                    im = Variable(im)
                    label = Variable(label)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                onesub_acc = get_acc(output, label)
                valid_acc += onesub_acc

                if epoch == 219:
                   _, predict_label = output.max(1)
                   single_sub_acc.append(onesub_acc)
                
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        if epoch == 219:
            print(epoch_str + time_str)
            # print(single_sub_acc)
    return single_sub_acc
        
        
if __name__ == '__main__':

    # fixed parameter
    blnum = 6
    sbnum = 35
    chnum = 9
    tarnum = 40
    harnum = 5

    # load data
    energy_distribution_data = np.load(r"9ch_3s_stdenergy.npy")
    allcross_singlesub_acc = np.zeros((blnum,sbnum))

    # cross validation
    for b_i in range(6):
        print('cross validation: %d' %(b_i+1))
        test_bl = b_i
        train_bl = np.setdiff1d(np.arange(blnum),b_i)
        
        traindata_array = np.zeros((35*5*40,harnum,chnum))
        i = 0
        for bp in range(5):
            for tar in range(40):
                for sub in range (35):
                    tempdate = energy_distribution_data[sub,train_bl[bp],:,tar]  
                    tempdata2 = tempdate.reshape(5,chnum)
                    traindata_array[i,...] = torch.tensor(tempdata2)
                    i = i+1
        traindata_tensor = torch.tensor(traindata_array)
        traindata_tensor = traindata_tensor.to(torch.float32)
        d = torch.arange(0, 35, 1)
        trainlabel = d.repeat(200)
        trainlabel = trainlabel
        # trainlabel = trainlabel.to(torch.float32)

        # data iteration
        train_set = subDataset(traindata_tensor, trainlabel)
        train_data = DataLoader(train_set, 64, True, num_workers=4)

        # test
        j = 0
        testdata_array = np.zeros((35*40,5,chnum))
        for sub in range (35):
            for tar in range(40):
                    tempdate3 = energy_distribution_data[sub,test_bl,:,tar]  
                    tempdata4 = tempdate3.reshape(5,chnum)
                    testdata_array[j,...] = torch.tensor(tempdata4)
                    j = j+1
        testdata_tensor = torch.tensor(testdata_array)
        testdata_tensor = testdata_tensor.to(torch.float32)
        d = torch.arange(0, 35, 1)
        testlabel = d.repeat_interleave(40)
        testlabel = testlabel
        # testlabel = testlabel.to(torch.float32)

        test_set = subDataset(testdata_tensor, testlabel)
        test_data = DataLoader(test_set, 40, False, num_workers=4)
        
        net = rnn_classify()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(net.parameters(), 1e-1)
        
        
        onecross_singlesub_acc = train(net, train_data, test_data, 220, optimizer, criterion)  
        allcross_singlesub_acc[b_i,:] = np.array(onecross_singlesub_acc) 
    np.save('acc_9ch_3s.npy',allcross_singlesub_acc)
    
        
    
    


