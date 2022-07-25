# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:29:44 2022

@author: aokis
House Prices - Advanced Regression Techniques
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
"""
import csv
import numpy as np
import pandas as pd
import itertools

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import statsmodels.api as sm
import statsmodels.regression.linear_model as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

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
        data[key+"_bin"] = data[key].where(data[key] < 1, 1)
    
    
    categoryVariable = [key for key in data if data[key].dtype == object]
    other_categoryVariable = ["2ndFlrSF_bin","EnclosedPorch_bin","3SsnPorch_bin","ScreenPorch_bin","PoolArea_bin","BedroomAbvGr", "BsmtFullBath", 
                              "BsmtHalfBath", "Fireplaces", "FullBath", "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold",
                              "MSSubClass", "OverallCond", "TotRmsAbvGrd", "YrSold"]
    
    if "SalePrice" in data.keys():
        data = standerdscaler(data, data.keys().drop(["SalePrice"] + categoryVariable + other_categoryVariable))
    else:
        data = standerdscaler(data, data.keys().drop(categoryVariable + other_categoryVariable))
        
    
    #カテゴリ変数をダミー変数に変換
    data = pd.get_dummies(data, columns=categoryVariable + other_categoryVariable, drop_first = True)
        
    return data

def standerdscaler(data, keys):
    "keysに与えられた変数を標準化して返す"
    standard_sc = StandardScaler()
    
    X = data.loc[:, keys]
    X = standard_sc.fit_transform(X)
    data.loc[:, keys] = X
    
    return data
    
def Training(data, targetVariable, UseTraingVariable):
    Y = data[targetVariable]
    x = data[UseTraingVariable]
    
    # 定数項(y切片)を必要とする線形回帰のモデル式ならば必須
    X = sm.add_constant(x)
    
    # 最小二乗法でモデル化
    model = smf.OLS(Y, X)
    result = model.fit()

    return result, model

def vif_calc(model):
    num_cols = model.exog.shape[1] # 説明変数の列数
    vifs = [vif(model.exog, i) for i in range(0, num_cols)]
    return pd.DataFrame(vifs, index=model.exog_names, columns=['VIF'])
    

def vif_u10_select(data ,result, vif_result):
    return list(vif_result[vif_result["VIF"] < 10].index)

def data_split(data, k):
    sliptIndexes = list(range(0,data.shape[0] + 1, int(data.shape[0]/k)))
    
    return [[pd.concat([data.iloc[:sliptIndexes[i], :],  data.iloc[sliptIndexes[i+1]:, :]]), 
             data.iloc[sliptIndexes[i]:sliptIndexes[i+1], :]] for i in range(k)]

def adj_r2_score(y_true, y_pred, p):
    return 1-(1-r2_score(y_true, y_pred)) * (len(y_true)-1) / (len(y_true) - p - 1)
    
def crossVaridation(data, responseVariable, useExplanatoryVariable, k):
    dataSetList = data_split(data, k)
    
    keyNum = len(useExplanatoryVariable)
    
    #総当たり
    #patterns = list(itertools.product(range(2), repeat=len(useExplanatoryVariable)))
    
    with open("result4_ols.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patternNo", "NumOfAdoptedVar", "mean PI"] +
                   ["PI " + str(i+1) for i in range(k)] +
                   useExplanatoryVariable)
        
        result, model = Training(data, responseVariable, useExplanatoryVariable)
        sortedAllExplanatoryVariablePvalue = result.pvalues[1:].sort_values(ascending=False)
        pattern = [0]*keyNum
        
        bestAveragePerformanceIndex = -np.Inf
        performanceIndexCollection = [0] * k
        
        for patternNo in range(keyNum):
            pattern[useExplanatoryVariable.index(sortedAllExplanatoryVariablePvalue.index[patternNo])] = 1
                
            UseTraingVariable = [useExplanatoryVariable[i] for i in range(keyNum) if pattern[i]]
            
            for i in range(k):
                result, model = Training(dataSetList[i][0], responseVariable, UseTraingVariable)
                
                Y_test = dataSetList[i][1][[responseVariable]]
                X_test = dataSetList[i][1][UseTraingVariable]
                
                X_test_const = pd.DataFrame({"const" :[1]*X_test.shape[0]})
                X_test_const.index = X_test.index
                X_test_const = X_test_const.join(X_test)
                
                predict = result.predict(X_test_const)
                
                performanceIndexCollection[i] = adj_r2_score(Y_test, predict, sum(pattern))
            
            patternMeanPerformanceIndex = np.mean(performanceIndexCollection)
            
            print(patternNo, end = "\t")
            for printObject in [patternMeanPerformanceIndex] + performanceIndexCollection:
                print("{:.3f}".format(printObject), end = "\t")
            print("")
            w.writerow([patternNo,sum(pattern),patternMeanPerformanceIndex] + performanceIndexCollection + pattern)
            
            if bestAveragePerformanceIndex < patternMeanPerformanceIndex:
                bestAveragePerformanceIndex = patternMeanPerformanceIndex
            else:
                pattern[useExplanatoryVariable.index(sortedAllExplanatoryVariablePvalue.index[patternNo])] = 0
            
    result, model = Training(data, responseVariable, UseTraingVariable)

    return result, model
    
def testData_predict(data, result):
    
    X_test_const = pd.DataFrame({"const" :[1]*data.shape[0]})
    X_test_const.index = data.index
    X_test_const = X_test_const.join(data[data.keys() & result.params.keys()[1:]])
    
    with open("submission4_ols.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "SalePrice"])
        
        pred = result.predict(X_test_const)
        for index in pred.index:
            
            w.writerow([index , pred[index]])

def main():
    #データを読み込む
    rawData_train, rawData_test = load_data()
    #学習データの前処理
    creansedData_train = data_creansing(rawData_train)
    creansedData_test = data_creansing(rawData_test)
    
    #result, model = Training(creansedData_train, "SalePrice", creansedData_train.keys().drop("SalePrice"))
    #vif_result = vif_calc(model)
    
    #vifU10_variable = vif_u10_select(creansedData_train, result, vif_result[1:])
    
    result, model = crossVaridation(creansedData_train, "SalePrice", list(set(creansedData_train.keys()) & set(creansedData_test.keys())), 5)
    
    #result, model = Training(standerdscale_train, "SalePrice", vifU10_corU05)
    
    testData_predict(creansedData_test, result)
    
    
if __name__ == "__main__":
    main()
