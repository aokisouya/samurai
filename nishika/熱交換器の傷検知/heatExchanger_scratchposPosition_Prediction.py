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

def data_read(filename):
    data = pd.read_csv(filename)
    data.rename(columns = {"Unnamed: 0": ""}, inplace=True)
    return data.set_index("")

def Training(data, labels):
    Y = labels["target"]
    X = data
    
    model = LogisticRegression(C = 1, random_state = 0) 
    model.fit(X, Y)
    
    return model

def test(data, model):
    with open("submission_1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "target"])
        
        pred = model.predict_proba(data)
        for index in range(len(pred)):
            
            w.writerow([data.index[index] , pred[index][1]])
    
    return
    
def main():
    
    labels = data_read("train_label.csv")
    creansed_data_train = data_read("train_data.csv")
    creansed_data_test = data_read("test_data.csv")
   
    model = Training(creansed_data_train, labels)
   
    test(creansed_data_test, model)
   
    return


if __name__ == "__main__":
    main()
    