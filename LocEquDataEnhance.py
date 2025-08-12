# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:29:21 2022

@author: ZJLAB
"""

# 1 import library
import numpy as np
import math 
import random
from scipy.stats import kendalltau


# 2 Define Position-equivalent class
'''
sn: sampling point number of one channel
fr: signal frequency
fs: sampling rate
trsod: agumentation number
'''
class GetPPESeq(object):
    def __init__(self,sn,fr,fs,trsod):
        self.SamNum = int(sn)
        self.fre = fr
        self.Fs = fs
        self.trsod = trsod
    
    def generateseq(self):
        OrSeq = np.arange(self.SamNum)
        CySam = math.floor(self.Fs / self.fre)
        CyNum = math.ceil(self.SamNum * self.fre / self.Fs)
        
        # Generate original sub vector
        OrSubVec = [[] for _ in range(CySam)]
        for i in np.arange(CySam):
            for j in np.arange(CyNum):
                if (round(j * self.Fs / self.fre)+i) < self.SamNum:
                    OrSubVec[i].append(round(j * self.Fs / self.fre)+i)
                else:
                    break
   
        # Reserve points that do not participate in the exchange 
        OrList = OrSeq.tolist()
        SetSubVec = set([i for item in OrSubVec for i in item])
        ReservePos = list(set(OrList).difference(SetSubVec))
        
        # Random disturb
        EquseqSet = np.array(OrSeq).reshape(1,len(OrSeq))
        while (EquseqSet.shape[0] < self.trsod):
            NewSubVec = []
            for i in np.arange(CySam):
                a = OrSubVec[i]
                random.shuffle(a)
                NewSubVec.append(a)
                
            # Generate a new complete sequence
            NewIncomList = []
            for i in np.arange(CyNum):
                for j in  np.arange(CySam):
                    if (len(NewSubVec[j])-1) < i:
                        break
                    else:
                        NewIncomList.append(NewSubVec[j][i])      
            # Merging sequence
            for x in ReservePos:
                NewIncomList.insert(x,x)
            
            # calculate Levenshtein distance 不知道选哪个评判标准，莱文思特距离也不合适
            Newseq = np.array(NewIncomList).reshape(1,len(NewIncomList))
            EquseqSet = np.vstack((EquseqSet,Newseq))
            
        # output result
        return EquseqSet
    
# 对数据进行变换
class ConstructSig(object):
    '''
    EquseqSet: np.array[m,n] one PPE seauence matrix for one target frequency
    Data: np.array[trial,channel,sample] one rawdata matrix   
    '''
    def __init__(self,EquseqSet,Data):
        self.EquseqSet = EquseqSet
        self.Data = Data
    
    
    # 所有导联用同一个序列进行变换
    def exchange(self):
        getrinum = self.EquseqSet.shape[0]
        # 针对多个trial
        if len(self.Data.shape)==3:
            NewData = np.zeros(self.Data.shape[0]*getrinum,self.Data.shape[1],self.Data.shape[2])
            for trial_i in range(self.Data.shape[0]):
                predata = self.Data[trial_i,...]
                NewData[trial_i*getrinum,...] = predata
                for seq_i in range(1,getrinum):
                    NewData[trial_i*getrinum+seq_i,...] = predata[:,self.EquseqSet[seq_i,:]]
        # 针对一个trial            
        else:
            NewData = np.zeros((getrinum,self.Data.shape[0],self.Data.shape[1]))
            NewData[0,...] = self.Data
            for seq_i in range(1,getrinum):
                NewData[seq_i,...] = self.Data[:,self.EquseqSet[seq_i,:]]
        return NewData 

    
    # 不同导联用不同序列进行变换  
    def diffexchange(self):
        getrinum = round(((self.EquseqSet.shape[0])-1)/3 + 1)
        # 针对一个trial进行变换  
        if len(self.Data.shape)==2:
            chnum = self.Data.shape[0]
            NewData_2 = np.zeros((getrinum,self.Data.shape[0],self.Data.shape[1]))
            for t_i in np.arange(getrinum):
                if t_i==0:
                    NewData_2[t_i,...] = self.Data
                else:
                    cut_sqset = self.EquseqSet[1+(t_i-1)*3:1+t_i*3,:]
                    complete_sqset = np.vstack((cut_sqset,cut_sqset,cut_sqset))
                    np.random.shuffle(complete_sqset)
                    for ch_i in np.arange(chnum):
                        prechdata = self.Data[ch_i,:].reshape(1,-1)
                        NewData_2[t_i,ch_i,:] = prechdata[:,complete_sqset[ch_i,:]]
        # 针对多个trial进行分析
        elif len(self.Data.shape)==3:
            trialnum = self.Data.shape[0]
            chnum = self.Data.shape[1]
            NewData_2 = np.zeros((getrinum*self.Data.shape[0], self.Data.shape[1], self.Data.shape[2]))
            for ortrial_j in np.arange(trialnum):
                for t_i in np.arange(getrinum):
                    if t_i==0:
                        NewData_2[ortrial_j*getrinum+t_i,...] = self.Data[ortrial_j,...]
                    else:
                        cut_sqset = self.EquseqSet[1+(t_i-1)*4:1+t_i*4,:]
                        complete_sqset = np.vstack((cut_sqset,cut_sqset,self.EquseqSet[0,:]))
                        np.random.shuffle(complete_sqset)
                        for ch_i in np.arange(chnum):
                            prechdata = self.Data[ortrial_j,ch_i,:].reshape(1,-1)
                            NewData_2[ortrial_j*getrinum+t_i,ch_i,:] = prechdata[:,complete_sqset[ch_i,:]]

        return NewData_2
            
            
                        
                        
                        
                    
                    
                
            
        
        
        
        
                    
        
            
            
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
    

                  