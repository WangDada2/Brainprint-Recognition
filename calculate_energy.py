# -*- coding: utf-8 -*-
"""
author: Lijie Wang

function: calculate space-frequency energy

date: 2023.04.25

condition: 9channels & 3s
"""

# 1 import python library
import sys
import numpy as np
import scipy.io as scio
from scipy import signal as sig
from LocEquDataEnhance import GetPPESeq, ConstructSig
from sklearn import preprocessing


# 2 set important parameter
# data collection parameter
Fs = 250
tarnum = 40
blnum = 6
sbnum = 35
maxwin = 3
N = int(maxwin * Fs)

# data analyzing parameter
Num_SN = 5
tarin_blnum = 1
neuroscan_ssvep_channel = np.array([48,54,55,56,57,58,61,62,63]) - 1           # 9导联，包括PO7和PO8
dur_shift = 0.5                                                                # duration for gaze shifting [s]
dur_delay = 0.14                                                               # visual latency being considered in the analysis [s]
num_channels = 9

# 3 Useful variables
fb_meter = [[(6,18),(2,22)],
            [(14,26),(10,30)],
            [(22,34),(18,38)],
            [(30,42),(26,46)],
            [(38,50),(34,54)],
            [(46,58),(42,62)],
            [(54,66),(50,70)]]
gp = [2,20]
                                                           
SSVEP_stim_freq = np.array([round(8 + i * 0.2, 1) for i in range(40)])         # array of stimulus frequencies
SSVEP_stim_freq = SSVEP_stim_freq.reshape((8,5))
SSVEP_stim_freq = SSVEP_stim_freq.T
SSVEP_stim_freq = SSVEP_stim_freq.flatten()
alpha_ci = 0.05                                                                # 100*(1-alpha_ci): confidence interval for accuracy                        
ci = 100 * (1 - alpha_ci)                                                      # confidence interval

# 4 Get path
# ssvep benchmark dataset collected by neuroscan from Tsinghua university  
MainPath = r"data\Benchmark\dataset"


# 5 generate CCA sine template
# 倍频数
multiple_freq = 5
# 参考信号时间
templ_time = 5
# 参考信号长度
templ_len = templ_time * Fs
# 正余弦参考信号
template_set = []
# 采样点
samp_point = np.linspace(0, (templ_len - 1) / Fs, int(templ_len), endpoint=True)
# (1 * 计算长度)的二维矩阵
samp_point = samp_point.reshape(1, len(samp_point))
for freq in SSVEP_stim_freq:
    # 基频 + 倍频
    test_freq = np.linspace(freq, freq * multiple_freq, int(multiple_freq), endpoint=True)
    # (1 * 倍频数量)的二维矩阵
    test_freq = test_freq.reshape(1, len(test_freq))
    # (倍频数量 * 计算长度)的二维矩阵
    num_matrix = 2 * np.pi * np.dot(test_freq.T, samp_point)
    cos_set = np.cos(num_matrix)
    sin_set = np.sin(num_matrix)
    cs_set = np.append(cos_set, sin_set, axis=0)
    template_set.append(cs_set)
    target_template_set = np.array(template_set)
    
     
# 6 Load data
#低通滤波器
Wp = 90 / (Fs / 2)
Ws = 100 / (Fs / 2)
N2, Wn2 = sig.buttord(Wp,Ws,3,30,analog=False,fs=None)
b, a = sig.butter(N2, Wn2, btype='lowpass',analog=False,fs=None)
trsod = 11
getrinum = round((trsod-1)/3 + 1)

stdchene = np.zeros((sbnum,blnum,Num_SN*num_channels,tarnum))
for sub in range(1,sbnum+1):
    if sub < 10:
        data_path = MainPath + '\\' + 'S0' + str(sub) + '.mat'
    else:
        data_path = MainPath + '\\' + 'S' + str(sub) + '.mat'
    eeg = scio.loadmat(data_path)
    data = eeg['data']
    
    constructed_sig = np.zeros((Num_SN,tarnum,num_channels,int(maxwin*Fs)))
    for bl_i in range(blnum):
        channelenergy = np.zeros((Num_SN,num_channels,tarnum))
        
        for tar_i in range(tarnum):
            # 10 data preprocessing
            allhcondata = np.zeros((5,8,num_channels,int(maxwin*Fs)))
            # condata_fredom = np.zeros((5,8,9,int(maxwin*Fs)))
            crop = np.arange(round(0.64*Fs), round((0.64+maxwin)*Fs)) 
            selcha = data[neuroscan_ssvep_channel,...]                         # cut 
            seltmpt = selcha[:,crop,:,:]
            
            # 滤波（频带为所在谐波-5 +5）
            # lowpass filter design -- Chebyshev type I filter design
            for h in np.arange(Num_SN):
                pref = SSVEP_stim_freq[tar_i] * (h+1) 
                Wp = [round(pref)-5, round(pref)+5]
                Ws = [round(pref)-9, round(pref)+9]
                gpass = 2
                gstop = 20
                N, Wn = sig.cheb1ord(Wp, Ws, gpass,gstop, fs=Fs)
                B1, A1 = sig.cheby1(N, 0.5, Wn, btype="bandpass", fs=Fs)
                bpdata = sig.filtfilt(B1, A1, seltmpt[:,:,tar_i,bl_i])
                
                # PPER  get test data transformed by pper
                SamPointNum = int(maxwin*Fs)
                getposequsq = GetPPESeq(SamPointNum, SSVEP_stim_freq[tar_i], Fs, trsod) 
                PosSq_list = getposequsq.generateseq() 
                excdata = ConstructSig(PosSq_list,bpdata)
                constructdata = excdata.exchange()
            
                # 计算增强信号的平均信号
                for i in np.arange(getrinum):
                    constructdata[i,...] = sig.filtfilt(B1, A1, constructdata[i,...])

                constructed_sig[h,tar_i,...] = np.mean(constructdata[1:getrinum,:,:],axis = 0)
                channelenergy[h,:,tar_i] = np.sum(np.power(constructed_sig[h,tar_i,...],2), axis = 1) 
                mim_max_scaler = preprocessing.MinMaxScaler()
                tempstd = mim_max_scaler.fit_transform(channelenergy[h,:,tar_i].reshape(num_channels,1))
                stdchene[sub-1,bl_i,h*num_channels:(h+1)*num_channels,tar_i] = tempstd.flatten()
        
# 存储数据
np.save('9ch_3s_stdenergy.npy',stdchene)
                
        