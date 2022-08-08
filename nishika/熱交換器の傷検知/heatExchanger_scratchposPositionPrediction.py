# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 09:37:59 2022

@author: aokis
"""
import glob
import csv

import pandas as pd
import numpy as np
import scipy.signal as sg

from sklearn.linear_model import LogisticRegression

def get_filelist(path):
    return [x.split("\\")[-1] for x in list(glob.glob(path + "*"))]

def metaData_loading():
    meta = pd.read_csv(".\\data\\train_meta_test.csv")
    #meta = pd.read_csv(".\\data\\train_meta.csv")
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

def traning_label(meta, data):
    
    labels = pd.DataFrame(columns = ["target"])
    
    for filename in meta.index:
        filename_segment = filename.split(".")[0]
        
        data_length = len(data[filename])
        scratchpos_position = np.zeros(data_length)
        
        for hoge in meta.loc[filename]:
            if type(hoge) == list:
                for i in range(0, len(hoge), 2):
                    scratchpos_position = one_patting(scratchpos_position, hoge[i], hoge[i+1])
                    
        for i in range(0, len(data[filename]), 50):
            if 1 in scratchpos_position[i:i+50]:
                record  = pd.Series({"target": 1},  name = filename_segment + "_" + str(i + 50))
            else:
                record  = pd.Series({"target": 0}, name = filename_segment + "_" + str(i + 50))
            labels = labels.append(record)
    
    return labels


    """        
        for 
             = sg.find_peaks(abs(data[filename].iloc[:,i]), height = 150)[0]
    """  

def data_creansing(data_list, data):
    peaks = [None]*8
    peaks_segment = [None]*8
    peaks_segment_data = [None]*8
    peakTypes = [None]*8
    
    columns = data[data_list[0]].columns
    creansed_data = pd.DataFrame(columns = columns)
    
    for filename in data_list:
        filename_segment = filename.split(".")[0]
        for no in range(8):
            peaks[no] = sg.find_peaks(abs(data[filename].iloc[:,no]), height = 150)[0]
            
        for i in range(0, len(data[filename]), 50):
            
            for no in range(8):
                peaks_segment[no] = peaks[no][peaks[no]>=i]
                peaks_segment[no] = peaks_segment[no][peaks_segment[no]<i+50]
                peaks_segment_data[no] = data[filename].iloc[:,no][peaks_segment[no]]
                
                if len(peaks_segment_data[no]) == 0:
                    peakTypes[no] = 0
                elif len(peaks_segment_data[no]) == sum(peaks_segment_data[no] >= 0):
                    peakTypes[no] = 1
                elif len(peaks_segment_data[no]) == sum(peaks_segment_data[no] < 0):
                    peakTypes[no] = 2
                else:
                    peakTypes[no] = 3
            
            record  = pd.Series({columns[0]: peakTypes[0], columns[1]: peakTypes[1],
                                 columns[2]: peakTypes[2], columns[3]: peakTypes[3],
                                 columns[4]: peakTypes[4], columns[5]: peakTypes[5],
                                 columns[6]: peakTypes[6], columns[7]: peakTypes[7]},  name = filename_segment + "_" + str(i + 50))
            
            creansed_data = creansed_data.append(record)
  
    creansed_data = pd.get_dummies(creansed_data, columns=columns, drop_first = True)
    return creansed_data

def Training(data, labels):
    Y = labels["target"]
    X = data
    
    model = LogisticRegression(C = 1, random_state = 0) 
    model.fit(X, Y)
    
    return model   

def test(data, model):
    with open("submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "target"])
        
        pred = model.predict(data)
        for index in range(len(pred)):
            
            w.writerow([data.index[index] , pred[index]])
    
    return
    
def main():
    print(1)
    meta = metaData_loading()
    train_data = data_loading(".\\data\\train\\", meta.index)
    
    test_data_filelist = get_filelist(".\\data\\test_test\\")
    test_data = data_loading(".\\data\\test_test\\", test_data_filelist)
    
    labels = traning_label(meta, train_data)
    creansed_data_train = data_creansing(meta.index, train_data)
    #creansed_data_train = data_creansing(train_data_filelist, test_data)
   
    model = Training(creansed_data_train, labels)
   
    test(test_data, model)
   
    return


if __name__ == "__main__":
    main()
    