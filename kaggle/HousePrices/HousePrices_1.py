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
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold

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
    
    #カテゴリ変数をダミー変数に変換
    data = pd.get_dummies(data, columns=categoryVariable + other_categoryVariable, drop_first = True)
        
    return data

def standerdscaler(data, keys):
    "keysに与えられた変数を標準化して返す"
    
    for key in keys:
        data[key] = (data[key] - data[key].mean()) / data[key].std(ddof=0)
        
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
    

def vif_corr(data ,result, vif_result):
    
    vif_u10 = vif_result[vif_result["VIF"] < 10]
    corr_value = abs(data.corr()["SalePrice"].drop("SalePrice"))
    
    return [key for key in vif_u10.index if corr_value[key]  > 0.3]

def crossVaridation(data, responseVariable, useExplanatoryVariable, k):
    sliptIndexes = list(range(0,data.shape[0], int(data.shape[0]/k)))
    
    keyNum = len(useExplanatoryVariable)
    
    #総当たり
    #patterns = list(itertools.product(range(2), repeat=len(useExplanatoryVariable)))
    
    #増加
    pattern = [0]*keyNum
    patterns = [[1] * n + [0] * (keyNum - n) for n in range(keyNum+1)]
    
    max_Adj_R_squared = 0
    
    with open("result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DataGroupID","patternNo", "R-squared", "Adj. R-squared"] + useExplanatoryVariable)
        
        print("DataGroupID\t","patternNo")
        for i in range(k-1):
            data_test = data.iloc[sliptIndexes[i]:sliptIndexes[i+1], :]
            data_train = pd.concat([data.iloc[:sliptIndexes[i], :],  data.iloc[sliptIndexes[i+1]:, :]])
          
            for patternNo in range(len(patterns)):
                print("%d\t" % i, patternNo)
                
                if 1 in patterns[patternNo]:
                    
                    UseTraingVariable = [useExplanatoryVariable[i] for i in range(keyNum) if patterns[patternNo][i]]
                    result, model = Training(data_train, responseVariable, UseTraingVariable)
                    
                    """
                    ## テストデータの精度の算出の実装がしたい！
                    Y_train = data_train[[responseVariable]]
                    X_train = data_train[UseTraingVariable]
                    
                    Y_test = data_test[[responseVariable]]
                    X_test = data_test[UseTraingVariable]
                    
                    X_test_const = pd.DataFrame({"const" :[1]*X_test.shape[0]})
                    X_test_const.index = X_test.index
                    X_test_const = X_test_const.join(X_test)
                    predict = result.predict(X_test_const)
                    
                    result.rsquared()
                    """
                    
                    if max_Adj_R_squared < result.rsquared_adj:
                        max_Adj_R_squared = result.rsquared_adj
                        max_Adj_R_squared_result, max_Adj_R_squared_model = result, model
                        
                    
                    w.writerow([i,patternNo,result.rsquared, result.rsquared_adj] + list(patterns[patternNo]))
                    
        for patternNo in range(len(patterns)):
            print("%d\t" % (k-1), patternNo)
            
            if 1 in patterns[patternNo]:
                
                UseTraingVariable = [useExplanatoryVariable[k-1] for i in range(keyNum) if patterns[patternNo][k-1]]
                result, model = Training(data_train, responseVariable, UseTraingVariable)
                
                """
                ## テストデータの精度の算出の実装がしたい！
                Y_train = data_train[[responseVariable]]
                X_train = data_train[UseTraingVariable]
                
                Y_test = data_test[[responseVariable]]
                X_test = data_test[UseTraingVariable]
                
                X_test_const = pd.DataFrame({"const" :[1]*X_test.shape[0]})
                X_test_const.index = X_test.index
                X_test_const = X_test_const.join(X_test)
                predict = result.predict(X_test_const)
                
                result.rsquared()
                """
                
                if max_Adj_R_squared < result.rsquared_adj:
                    max_Adj_R_squared = result.rsquared_adj
                    max_Adj_R_squared_result, max_Adj_R_squared_model = result, model
                
                w.writerow([i,patternNo,result.rsquared, result.rsquared_adj] + list(patterns[patternNo]))

    return max_Adj_R_squared_result, max_Adj_R_squared_model
    

def testData_predict(data, result):
    
    X_test_const = pd.DataFrame({"const" :[1]*data.shape[0]})
    X_test_const.index = data.index
    X_test_const = X_test_const.join(data[result.params.keys()[1:]])
    
    with open("submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "SalePrice"])
        
        pred = result.predict(X_test_const)
        for index in pred.index:
            
            w.writerow([index , pred[index]])

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
    
def main():
    #データを読み込む
    rawData_train, rawData_test = load_data()
    #学習データの前処理
    creansedData_train = data_creansing(rawData_train)
    creansedData_test = data_creansing(rawData_test)
    
    standerdscale_train = standerdscaler(creansedData_train, creansedData_train.keys().drop("SalePrice"))
    standerdscale_test = standerdscaler(creansedData_test, creansedData_test.keys())
    
    result, model = Training(standerdscale_train, "SalePrice", creansedData_train.keys().drop("SalePrice"))
    vif_result = vif_calc(model)
    
    vifU10_corU05 = vif_corr(standerdscale_train, result, vif_result[1:])
    
    result, model = crossVaridation(standerdscale_train, "SalePrice", vifU10_corU05, 5)
    
    #result, model = Training(standerdscale_train, "SalePrice", vifU10_corU05)
    
    testData_predict(standerdscale_test, result)
    
    return result, vif_result, vifU10_corU05
    #result = test(model, creansedData_test, CompletedData_test, creansedData_train.keys()[1:],  "result.csv")
    
    
if __name__ == "__main__":
    result, vif_result, useVar = main()
