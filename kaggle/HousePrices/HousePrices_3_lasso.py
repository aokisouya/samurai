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
from sklearn.linear_model import Lasso

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
    X = data[UseTraingVariable]
    
    # LASSO回帰
    lasso = Lasso(alpha=0.1)
    
    # モデル学習
    lasso.fit(X, Y)

    return lasso

def vif_calc(model):
    num_cols = model.exog.shape[1] # 説明変数の列数
    vifs = [vif(model.exog, i) for i in range(0, num_cols)]
    return pd.DataFrame(vifs, index=model.exog_names, columns=['VIF'])
    

def vif_corr(data ,result, vif_result):
    
    vif_u10 = vif_result[vif_result["VIF"] < 10]
    corr_value = abs(data.corr()["SalePrice"].drop("SalePrice"))
    
    return [key for key in vif_u10.index if corr_value[key] > 0.1]

def crossVaridation(data, responseVariable, useExplanatoryVariable, k):
    sliptIndexes = list(range(0,data.shape[0], int(data.shape[0]/k)))
    
    keyNum = len(useExplanatoryVariable)
    
    #総当たり
    #patterns = list(itertools.product(range(2), repeat=len(useExplanatoryVariable)))
    
    with open("result3_lasso.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DataGroupID","patternNo", "r2_train", "r2"] + useExplanatoryVariable)
        
        best_r2 = 0
        
        #sortedAllExplanatoryVariablePvalue = result.pvalues[1:].sort_values(ascending=False)
        sortedAllExplanatoryCorr = abs(data[[responseVariable] + useExplanatoryVariable].corr()["SalePrice"].drop("SalePrice")).sort_values(ascending=False)
        
        print("DataGroupID\t","patternNo")
        for i in range(k-1):
            data_test = data.iloc[sliptIndexes[i]:sliptIndexes[i+1], :]
            data_train = pd.concat([data.iloc[:sliptIndexes[i], :],  data.iloc[sliptIndexes[i+1]:, :]])
            
            pattern = [0]*keyNum
            gurup_best_r2 = 0
            
            for patternNo in range(keyNum):
                
                pattern[sortedAllExplanatoryCorr.index.get_loc(useExplanatoryVariable[patternNo])] = 1
                    
                UseTraingVariable = [useExplanatoryVariable[i] for i in range(keyNum) if pattern[i]]
                model = Training(data_train, responseVariable, UseTraingVariable)
                
                """
                ## テストデータの精度の算出の実装がしたい！
                """
                Y_train = data_train[[responseVariable]]
                X_train = data_train[UseTraingVariable]
                
                Y_test = data_test[[responseVariable]]
                X_test = data_test[UseTraingVariable]
                
                predict = model.predict(X_test)
                
                r2_train = r2_score(Y_train, model.predict(X_train))
                r2 = r2_score(Y_test, predict)
                print("%d\t" % i, patternNo, r2)
                
                if gurup_best_r2 < r2:
                    gurup_best_r2 = r2
                else:
                    pattern[patternNo] = 0
                    
                if best_r2 < r2:
                    best_r2 = r2
                    best_model = model
                    best_model_Variable = UseTraingVariable
                    
                w.writerow([i,patternNo,r2_train,r2] + pattern)
        
        data_test = data.iloc[sliptIndexes[i+1]:, :]
        data_train = data.iloc[:sliptIndexes[i+1], :]
        pattern = [0]*keyNum
        gurup_best_r2 = 0
        
        sortedAllExplanatoryCorr = abs(data[[responseVariable] + useExplanatoryVariable].corr()["SalePrice"].drop("SalePrice")).sort_values(ascending=False)
         
        for patternNo in range(keyNum):
            
            pattern[sortedAllExplanatoryCorr.index.get_loc(useExplanatoryVariable[patternNo])] = 1
        
            UseTraingVariable = [useExplanatoryVariable[i] for i in range(keyNum) if pattern[i]]
            model = Training(data_train, responseVariable, UseTraingVariable)
            
            """
            ## テストデータの精度の算出の実装がしたい！
            """
            Y_train = data_train[[responseVariable]]
            X_train = data_train[UseTraingVariable]
            
            Y_test = data_test[[responseVariable]]
            X_test = data_test[UseTraingVariable]
            
            predict = model.predict(X_test)
            
            r2_train = r2_score(Y_train, model.predict(X_train))
            r2 = r2_score(Y_test, predict)
            
            print("%d\t" % (k-1), patternNo, r2)
        
            if gurup_best_r2 < r2:
                gurup_best_r2 = r2
            else:
                pattern[patternNo] = 0
                
            if best_r2 < r2:
                best_r2 = r2
                best_model = model
                best_model_Variable = UseTraingVariable
            
            w.writerow([i,patternNo,r2_train,r2] + pattern)
            
    print("best_r2", best_r2)

    return best_model, best_model_Variable
    
def testData_predict(data, model, UseTraingVariable):
    with open("submission3_lasso.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id", "SalePrice"])
        
        pred = model.predict(data[UseTraingVariable])
        for index in range(len(pred)):
            
            w.writerow([data.index[index] , pred[index]])

    
def main():
    #データを読み込む
    rawData_train, rawData_test = load_data()
    #学習データの前処理
    creansedData_train = data_creansing(rawData_train)
    creansedData_test = data_creansing(rawData_test)
    
    #result, model = Training(creansedData_train, "SalePrice", creansedData_train.keys().drop("SalePrice"))
    #vif_result = vif_calc(model)
    
    #vifU10_corU05 = vif_corr(creansedData_train, result, vif_result[1:])
    
    model, UseTraingVariable = crossVaridation(creansedData_train, "SalePrice", list(set(creansedData_train) & set(creansedData_test.keys())), 5)
    
    #result, model = Training(standerdscale_train, "SalePrice", vifU10_corU05)
    
    testData_predict(creansedData_test, model, UseTraingVariable)
    
    #return result, vif_result, vifU10_corU05
    #result = test(model, creansedData_test, CompletedData_test, creansedData_train.keys()[1:],  "result.csv")
    
    
if __name__ == "__main__":
    main()
