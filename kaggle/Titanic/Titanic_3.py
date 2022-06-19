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
from sklearn.linear_model import LogisticRegression
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
    rawData_train = pd.read_csv(r".\rawdata\train.csv")
    rawData_test = pd.read_csv(r".\rawdata\test.csv")
    return rawData_train, rawData_test

      
def missingValue_completion(data):
    """
    データの欠損値を補完する
    """

    #年齢の欠損を最頻値で埋める
    #ToDo　データを分割する前に欠損を埋めるべき
    data["Age"] = data["Age"].fillna(data["Age"].mode()[0])
    #乗船港の欠損を最頻値で埋める
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    
    return data
        
def Undersampling(data):
    
    data = data.sample(frac=1, random_state=0)
    SurvivedSum = data["Survived"].sum()
    dataLen = data.shape[0]
    counter = 0
    if SurvivedSum < dataLen - SurvivedSum:
        for i in data.index:            
            if data["Survived"][i] == 0:
                if counter < SurvivedSum:
                    counter += 1
                else:
                    data = data.drop(index = i)
    else:
        for i in data.index:
            if data["Survived"][i] == 1:
                if counter < dataLen - SurvivedSum:
                    counter += 1
                    print(counter)
                else:
                    data = data.drop(index = i)
                    
    return data

def data_creansing(rawData):
    """
    
    """
    #PassengerIdをIndexに変換
    creansedData = rawData.set_index("PassengerId", verify_integrity = True)
    
    # 年齢を15歳刻みでまとめる
    creansedData["Age"] = round(creansedData["Age"] / 15) * 15
    #性別と年齢の目的変数をまとめる
    creansedData["SexAge"] = creansedData["Sex"] + "-" + creansedData["Age"].astype(str)
    
    #運賃を50刻みでまとめる
    creansedData["Fare"] = round(creansedData["Fare"] / 50) * 50
    
    #Sex-Ageについて生存率で3段階に分ける
    creansedData = creansedData.replace({"SexAge": ['male-15.0', 'male-30.0', 'male-45.0', 'male-60.0']}, "Low")
    creansedData = creansedData.replace({"SexAge": ['male-0.0','female-0.0', 'female-15.0', 'female-30.0', 'female-45.0']}, "Middle")
    creansedData = creansedData.replace({"SexAge": ['female-60.0', 'male-75.0' ]}, "High")
    
    #Ticketを数字だけとそれ以外のチケットでカテゴリ変数に変換
    #チケット情報を文字列の先頭が「数値かそれ以外か」を条件に数字のみのチケットかそれ以外に分類：bool
    numberOnlyTicket = creansedData["Ticket"].str[0].isin(map(str, list(range(10))))
    creansedData["Ticket"] = creansedData["Ticket"].where(~numberOnlyTicket,"number")
    creansedData["Ticket"] = creansedData["Ticket"].where(numberOnlyTicket,"Other than number")
    
    #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
    #Cabinの情報にて、nanを客室未割当と定義した後、生存率で3段階に分ける
    creansedData["Cabin"] = creansedData["Cabin"].str[0]
    creansedData["Cabin"] = creansedData["Cabin"].where(~creansedData["Cabin"].isnull(),"Low")
    creansedData = creansedData.replace({"Cabin": ["T", "N"]},"Low")
    creansedData = creansedData.replace({"Cabin": ["A","G","C","F"]},"Middle")
    creansedData = creansedData.replace({"Cabin": ["B","E","D"]},"high")
    
    #SibSpを1,2.3,4以上でカテゴリ変数へ変換
    creansedData["SibSp"] = creansedData["SibSp"].where(creansedData["SibSp"] < 4, 4)
    
    #Parchを1,2.3,4,5以上でカテゴリ変数へ変換
    creansedData["Parch"] = creansedData["Parch"].where(creansedData["Parch"] < 5, 5)
    
    
    #Sex-Ageについて生存率で3段階に分ける
    creansedData = creansedData.replace({"Fare": [0, 300, 350, 400, 450]}, "Low")
    creansedData = creansedData.replace({"Fare": [50, 100, 150, 200, 250]}, "Middle")
    creansedData = creansedData.replace({"Fare": [500]}, "High")
    
    #Undersampling 
    """
    生存と死亡のデータ数を揃える
    crossValidation を行う際に生存と死亡の比率が変わるため、意味が薄れてしまうと感じた為一度スキップ
    if "Survived" in creansedData.keys():
        creansedData = Undersampling(creansedData)
    """
    
    #使わない説明変数削除 
    creansedData = creansedData.drop(columns = ["Name", "Age", "Sex"] , axis=1)
    
    #カテゴリ変数をダミー変数に変換
    creansedData = pd.get_dummies(creansedData, columns=["SexAge","SibSp","Parch","Ticket", "Cabin", "Embarked", "Pclass", "Fare"], drop_first = True)
    
    #(修正箇所)欠損のあるデータを中央値で補完
    #!!!最頻値で補完に修正する
    #ダミー変数も一緒にやっていいかはよく分かっていない　簡単に検索したけど見つからなかった
    #中央値ならダミーにも0/1以外入る事は稀のはずなのでみためも気持ち悪さもあまりない
    creansedData = creansedData.fillna(creansedData.median())
    
    return creansedData

def Training(data, targetVariable, UseTraingVariable):
    Y = data[targetVariable]
    X = data[UseTraingVariable]
    
    model = LogisticRegression(C = 1, random_state = 0) 
    model.fit(X, Y)
    
    return model

def CrossValidation(data, targetVariable, k):
    
    logistic = smf.glm(formula = "Survived ~ SexAge_Low + SexAge_Middle + SibSp_1 + SibSp_2 + SibSp_3 + SibSp_4 + Parch_1 + Parch_2 + Parch_3 + Parch_4 + Parch_5 + Ticket_number + Cabin_Middle + Cabin_high + Embarked_Q + Embarked_S + Pclass_2 + Pclass_3 + Fare_Low + Fare_Middle",
                   data = data,
                   family = sm.families.Binomial()).fit()
    print(logistic.summary())
    
    sliptIndexes = list(range(0,data.shape[0], int(data.shape[0]/k)))
    
    ExplanatoryVariable = list(data.keys().drop(targetVariable))
    
    keyNum = len(ExplanatoryVariable)
    
    patterns = list(itertools.product(range(2), repeat=len(ExplanatoryVariable)))
    
    with open("result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DataGroupID","patternNo","Survived"] + ExplanatoryVariable)
        for i in range(k-1):
            data_test = data.iloc[sliptIndexes[i]:sliptIndexes[i+1], :]
            data_train = pd.concat([data.iloc[:sliptIndexes[i], :],  data.iloc[sliptIndexes[i+1]:, :]])
          
            for patternNo in range(len(patterns)):
                if 1 in patterns[patternNo]:
                    UseTraingVariable = [ExplanatoryVariable[i] for i in range(keyNum) if patterns[patternNo][i]]
                    Y = data[[targetVariable]]
                    X = data[UseTraingVariable]
                    model = Training(data, targetVariable, UseTraingVariable)
                    
                    w.writerow([i,patternNo,format(model.score(X, Y))] + list(patterns[patternNo]))
                   
        
    data_test = data.iloc[sliptIndexes[k-1]:, :]
    data_train = data.iloc[:sliptIndexes[k-1], :]
        
    return

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
    
    index = list(rawData["PassengerId"])
    
    with open("./" + resultFileName, "w", newline="") as f:
        w = csv.writer(f)
        if flag == 0:
            w.writerow(["PassengerId","Survived"])
            for i in range(len(result)):
                w.writerow([index[i], result[i]])
        elif flag == 1:
            count = 0
            correct = 0
            w.writerow(["PassengerId","Survived", "answer"])
            for i in range(len(result)):
                count += 1
                if result[i] == answer[i]:
                    correct += 1
                w.writerow([index[i], result[i], answer[i]])
            
            print("正解率", correct / count)
            
    return result
    
def main():
    #cross validation で分割する個数
    k = 5
    
    #データを読み込む
    rawData_train, rawData_test = load_data()
    #欠損したデータを補完
    CompletedData_train = missingValue_completion(rawData_train)
    CompletedData_test = missingValue_completion(rawData_test)
    
    #学習データの前処理
    creansedData_train = data_creansing(CompletedData_train)
    creansedData_test = data_creansing(CompletedData_test)
    
    
    #crossVaridation test
    CrossValidation(creansedData_train, "Survived", k)

    
    
    #result = test(model, creansedData_test, CompletedData_test, creansedData_train.keys()[1:],  "result.csv")
    
if __name__ == "__main__":
    main()   
    