# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:29:44 2022

@author: aokis
Titanic - Machine Learning from Disaster
https://www.kaggle.com/competitions/titanic
"""
import csv
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold

import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def load_data():
    """
    ファイルからデータを読み込む
    """
    rawData_train = pd.read_csv(r".\data\train.csv")
    rawData_test = pd.read_csv(r".\data\test.csv")
    return rawData_train, rawData_test

def data_creansing(data):
    """
    
    """
    #PassengerIdをIndexに変換
    data = data.set_index("Id", verify_integrity = True)

    #使わない説明変数削除 
    data = data.drop(columns = ["BsmtFinSF1", "BsmtFinSF2", "LowQualFinSF"] , axis=1)
        
    #欠損を最頻値で補完
    for key in data.keys():
        data[key] = data[key].fillna(data[key].mode()[0])
    
    #0かそれ以外でカテゴリに変換
    binaryVariable = ["2ndFlrSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea"]
    
    for key in binaryVariable:
        data[key] = data[key].where(data[key] < 1, 1)
    
    
    categoryVariable = [key for key in data if data[key].dtype == object]
    other_categoryVariable = ["2ndFlrSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","BedroomAbvGr", "BsmtFullBath", 
                              "BsmtHalfBath", "Fireplaces", "FullBath", "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold",
                              "MSSubClass", "OverallCond", "TotRmsAbvGrd", "YrSold"]
    
    #カテゴリ変数をダミー変数に変換
    data = pd.get_dummies(data, columns=categoryVariable + other_categoryVariable, drop_first = True)
        
    return data

def Training(data, targetVariable, UseTraingVariable):
    Y = data[targetVariable]
    X = data[UseTraingVariable]
    
    model = LinearRegression() 
    model.fit(X, Y)
    
    return model



def test(model, data, rawData, keys, resultFileName):
    """
    メモ        

    """
    flag = 0
    
    if "Survived" in rawData.keys():
        answer = list(rawData["Survived"])
        flag = 1
    
    #足りない変数を足しています。
    for key in keys:
        if key not in data.keys():
            data.insert(loc = 0, column = key, value = 0)

    data = data.reindex(columns = keys)
    
    #bias = model.intercept_
    #weight = model.coef_
    
    result = model.predict(data)
    
    index = list(rawData["Id"])
    
    with open("./" + resultFileName, "w", newline="") as f:
        w = csv.writer(f)
        if flag == 0:
            w.writerow(["PassengerId","Survived"])
            for i in range(len(result)):
                w.writerow([index[i], result[i]])
        elif flag == 1:
            count = 0
            correct = 0
            w.writerow(["Id","Survived", "answer"])
            for i in range(len(result)):
                count += 1
                if result[i] == answer[i]:
                    correct += 1
                w.writerow([index[i], result[i], answer[i]])
            
            print("正解率", correct / count)
            
    return result
    
def main():\
    #データを読み込む
    rawData_train, rawData_test = load_data()
    #学習データの前処理
    creansedData_train = data_creansing(rawData_train)
    creansedData_test = data_creansing(rawData_test)
    
    
    model = Training(creansedData_train, "SalePrice", creansedData_train.keys().drop("SalePrice"))
    
    #crossVaridation test

    
    return
    #result = test(model, creansedData_test, CompletedData_test, creansedData_train.keys()[1:],  "result.csv")
    
    
if __name__ == "__main__":
    main()   
    