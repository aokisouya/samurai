# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 09:37:59 2022

@author: aokis
"""
import glob
import csv
from concurrent import futures

import pandas as pd
import numpy as np
import scipy.signal as sg

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_filelist(path):
    return [x.split("\\")[-1] for x in list(glob.glob(path + "*"))]

def metaData_loading(path):
    meta = pd.read_csv(path)
    meta = meta.set_index("filename")
    meta = meta.applymap(lambda x: list(map(int, x.split(" ")[:-1])) if type(x) is str and " " in x else x)
    
    return meta
    
def data_loading(path, filename_list):
    data = {}
    
    for filename in filename_list:
        data[filename] =  pd.read_csv(path + filename)
    
    return data
 
def one_patting(data, s, e):
    data[s:e] = 1
    return data

def onefile_label(meta, data_length, filename, n):
    print("label thread {:04}: Start".format(n))
    labels = pd.DataFrame(columns = ["target"])
    filename_segment = filename.split(".")[0]
    
    scratchpos_position = np.zeros(data_length)
    
    for hoge in meta:
        if type(hoge) == list:
            for i in range(0, len(hoge), 2):
                scratchpos_position = one_patting(scratchpos_position, hoge[i], hoge[i+1])
                
    for i in range(0, data_length-50, 50):
        if 1 in scratchpos_position[i:i+50]:
            record  = [1]
        else:
            record  = [0]
        labels.loc[filename_segment + "_" + str(i+50), :] = record
    
    print("label thread {:04}: End".format(n))
    
    return labels

def traning_label(meta, data):
    data_list = meta.index
    labels = pd.DataFrame(columns = ["target"])
    
    
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        future = [executor.submit(onefile_label, 
                                    meta = meta.loc[data_list[i]] ,data_length = len(data[data_list[i]]), 
                                    filename = data_list[i], n = i) for i in range(len(data_list))]

    list_index = list(range(len(data_list)))
    
    for i in range(len(data_list)):
        for ii in list_index:
            if future[ii].result().index[0][:-3] == data_list[i][:-4]:
                labels = pd.concat([labels, future[ii].result()])
                list_index.remove(i)
                continue
    
    labels = labels.astype("int8")
    return labels

def rms(data):
    return 

def feature_extraction(data, filename, columns, n):
    print("feature thread {:04}: Start".format(n))
    
    creansed_data_onefile = pd.DataFrame(columns = columns)
    
    peaks_point = [None]*8
    limited_peaks = [None]*8
    peaks_segment = [None]*8
    peaks_segment_data = [None]*8
    peakTypes = [None]*8
    average = [None]*8
    variance = [None]*8
    RMS = [None]*8
    maxValue = [None]*8
    #peakNum  = [None]*8
    
    peal_threshold = [10,10,10,10,5,5,10,10]
    peak_limit = [250, 1700, 500, 300, 150, 100, 250, 1000]
    #bandpass firter
    N = len(data)
    dt = 0.001
    
    filename_segment = filename.split(".")[0]
    
    for no in range(8):
        bandpass_filter = sg.firwin(10, cutoff = [50, 100], width=None, 
                    window='hamming', pass_zero=False, fs=1/dt)
        
        peak_detection_data = data.iloc[:,no] - np.median(data.iloc[:,no])
        
        if not peak_detection_data[peak_detection_data>=1000].empty:
            peak_detection_data[peak_detection_data>=1000] = 0
        if not peak_detection_data[peak_detection_data<=-1000].empty:
            peak_detection_data[peak_detection_data<=-1000] = 0
        peak_detection_data = sg.lfilter(bandpass_filter, 1, peak_detection_data)
        
        peak_detection_data = np.gradient(peak_detection_data)
        
        symbol = (peak_detection_data <0)*-1
        symbol[symbol == 0] = 1
        
        peak_detection_data = peak_detection_data - np.median(peak_detection_data)
        peaks_point[no] = sg.find_peaks(abs(peak_detection_data), height = peal_threshold[no])[0]
        
        limited_peaks[no] = peaks_point[no][np.where(peak_detection_data[peaks_point[no]] < peak_limit[no])[0]]
        limited_peaks[no] = peaks_point[no][np.where(peak_detection_data[limited_peaks[no]] > -peak_limit[no])[0]]
        
    data_len = len(data)
    for i in range(0, data_len-50, 50):

        for no in range(8):
            peaks_segment[no] = limited_peaks[no][limited_peaks[no]>=i]
            peaks_segment[no] = peaks_segment[no][peaks_segment[no]<i+50]
            peaks_segment_data[no] = data.iloc[:,no][peaks_segment[no]]
            
            if len(peaks_segment_data[no]) == 0:
                peakTypes[no] = 0
            elif len(peaks_segment_data[no]) == sum(peaks_segment_data[no] >= 0):
                peakTypes[no] = 1
            elif len(peaks_segment_data[no]) == sum(peaks_segment_data[no] < 0):
                peakTypes[no] = 2
            else:
                peakTypes[no] = 3
            
            data_segment = data.iloc[i:i+50,no]
            
            average[no] = np.mean(np.absolute(data_segment))
            variance[no] = np.var(data_segment)
            RMS[no] = np.sqrt(np.square(data_segment).mean(axis=0))
            maxValue[no] = max(data_segment)
            #peakNum[no] = len(peaks[no])
            
        creansed_data_onefile.loc[filename_segment + "_" + str(i+50), :] = peakTypes +average + variance + RMS + maxValue
    
    print("feature thread {:04}: End".format(n))
        
    return creansed_data_onefile 

def standerdscaler(data, keys):
    "keysに与えられた変数を標準化して返す"
    standard_sc = StandardScaler()
    
    X = data.loc[:, keys]
    X = standard_sc.fit_transform(X)
    data.loc[:, keys] = X
    
    return data

def data_creansing(data_list, data):
    
    #作成する変数の行名を作成
    columns = data[data_list[0]].columns
    columns_1 = [x+"-peakType" for x in columns] + [x+"-average" for x in columns] + [x+"-variance" for x in columns] + [x+"-RMS" for x in columns] + [x+"-maxPeak" for x in columns]
    creansed_data = pd.DataFrame(columns = columns_1)
    #ダミー変数にする行を指定
    dummyvar = [x+"-peakType" for x in columns]
    #int型を宣言する行を指定
    intvar = [x+"-peakType" for x in columns]
    intvar = dict(zip(intvar, ["int8"] * len(intvar)))
    #float型を宣言する行を指定
    floatvar = [x+"-average" for x in columns] + [x+"-variance" for x in columns] + [x+"-RMS" for x in columns] + [x+"-maxPeak" for x in columns]
    floatvar = dict(zip(floatvar, ["float32"] * len(floatvar)))
    
    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        future = [executor.submit(feature_extraction, 
                                 data = data[data_list[i]], filename = data_list[i], 
                                 columns = columns_1, n = i) for i in range(len(data_list))]
            
    list_index = list(range(len(data_list)))
    
    for i in range(len(data_list)):
        for ii in list_index:
            if future[ii].result().index[0][:-3] == data_list[i][:-4]:
                creansed_data = pd.concat([creansed_data, future[ii].result()])
                list_index.remove(i)
                continue      
    
    creansed_data = creansed_data.astype(intvar)
    creansed_data = creansed_data.astype(floatvar)

    #!!!正規化
    creansed_data = standerdscaler(creansed_data, list(floatvar.keys()))
    
    creansed_data = pd.get_dummies(creansed_data, columns=dummyvar, drop_first = True)
    
    return creansed_data

def to_csv(data, filename):
    data.to_csv(filename)

    
def main():
    print("metaData_loading")
    meta_file = ".\\data\\train_meta.csv"
    #meta_file = ".\\data\\train_meta_test.csv"
    meta = metaData_loading(meta_file)
    print("trainData_loading")
    train_data = data_loading(".\\data\\train\\", meta.index)
    
    print("get testData list")
    path = ".\\data\\test\\"
    #path = ".\\data\\test_test\\"
    test_data_filelist = get_filelist(path)
    print("testData_loading")
    test_data = data_loading(path, test_data_filelist)
    
    print("get label")
    labels = traning_label(meta, train_data)
    print("trainData_creanse")
    creansed_data_train = data_creansing(meta.index, train_data)
    print("testData_creanse")
    creansed_data_test = data_creansing(test_data_filelist, test_data)
    
    to_csv(labels, "train_label.csv")
    to_csv(creansed_data_train, "train_data.csv")
    to_csv(creansed_data_test, "test_data.csv")
   
    return


if __name__ == "__main__":
    main()
    