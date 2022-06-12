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
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def load_data():
    """
    Returns
    -------
    TYPE
        DataFrame

    """
    rawData_train = pd.read_csv(r".\rawdata\train.csv")
    rawData_test = pd.read_csv(r".\rawdata\test.csv")
    return rawData_train, rawData_test

def validation(data):
    return train_test_split(data,         # 訓練データとテストデータに分割する
                     test_size=0.3,       # テストデータの割合
                     shuffle=True,        # シャッフルする
                     random_state=0)      # 乱数シードを固定する
        
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
    
    #年齢の欠損を最頻値で埋める
    creansedData["Age"] = creansedData["Age"].fillna(creansedData["Age"].mode()[0])
    #乗船港の欠損を最頻値で埋める
    creansedData["Embarked"] = creansedData["Embarked"].fillna(creansedData["Embarked"].mode()[0])
    
    # 年齢を15歳刻みでまとめる
    creansedData["Age"] = round(creansedData["Age"] / 15) * 15
    #性別と年齢の目的変数をまとめる
    creansedData["Sex-Age"] = creansedData["Sex"] + "-" + creansedData["Age"].astype(str)
    
    #運賃を50刻みでまとめる
    creansedData["Fare"] = round(creansedData["Fare"] / 50) * 50
    
    #Sex-Ageについて生存率で3段階に分ける
    creansedData = creansedData.replace({"Sex-Age": ['male-15.0', 'male-30.0', 'male-45.0', 'male-60.0']}, "Low")
    creansedData = creansedData.replace({"Sex-Age": ['male-0.0','female-0.0', 'female-15.0', 'female-30.0', 'female-45.0']}, "Middle")
    creansedData = creansedData.replace({"Sex-Age": ['female-60.0', 'male-75.0' ]}, "High")
    
    #Ticketを数字だけとそれ以外のチケットでカテゴリ変数に変換
    #チケット情報を文字列の先頭が「数値かそれ以外か」を条件に数字のみのチケットかそれ以外に分類：bool
    numberOnlyTicket = creansedData["Ticket"].str[0].isin(map(str, list(range(10))))
    creansedData["Ticket"] = creansedData["Ticket"].where(~numberOnlyTicket,"number")
    creansedData["Ticket"] = creansedData["Ticket"].where(numberOnlyTicket,"Other than number")
    
    
    #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
    #Cabinの情報にて、nanを客室未割当と定義した後、生存率で3段階に分ける
    creansedData["Cabin"] = creansedData["Cabin"].where(~creansedData["Cabin"].isnull(),"Low")
    creansedData = creansedData.replace({"Cabin": ["T"]},"Low")
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
    if "Survived" in creansedData.keys():
        creansedData = Undersampling(creansedData)
    
    #使わない説明変数削除 
    creansedData = creansedData.drop(columns = ["Name", "Age", "Sex"] , axis=1)
    
    #カテゴリ変数をダミー変数に変換
    creansedData = pd.get_dummies(creansedData, columns=["Sex-Age","SibSp","Parch","Ticket", "Cabin", "Embarked", "Pclass", "Fare"], drop_first = True)
    
    #(修正箇所)欠損のあるデータを中央値で補完
    #!!!最頻値で補完に修正する
    #ダミー変数も一緒にやっていいかはよく分かっていない　簡単に検索したけど見つからなかった
    #中央値ならダミーにも0/1以外入る事は稀のはずなのでみためも気持ち悪さもあまりない
    creansedData = creansedData.fillna(creansedData.median())
    
    return creansedData


def Training(data):
    Y = data["Survived"]
    X = data[[Variable for Variable in data.keys() if Variable != "Survived"]]
    
    model = LogisticRegression(C = 1, random_state = 0) 
    model.fit(X, Y)
    
    train_score = format(model.score(X, Y))
    print('正解率(train):', train_score)
    
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
    
    rawData_train, rawData_test = load_data()
    rawData_train_train, rawData_train_test = validation(rawData_train)

    creansedData_train_train = data_creansing(rawData_train_train)
    creansedData_train_test = data_creansing(rawData_train_test.drop("Survived", axis = 1))
    creansedData_test = data_creansing(rawData_test)
    
    model = Training(creansedData_train_train)
    
    #return rawData_train, rawData_train_train, rawData_train_test, rawData_test, creansedData_train_train, creansedData_train_test, creansedData_test, model, None, None

 
    result_test = test(model, creansedData_train_test, rawData_train_test, creansedData_train_train.keys()[1:], "result_test.csv")
    result = test(model, creansedData_test, rawData_test, creansedData_train_train.keys()[1:],  "result.csv")
    return rawData_train, rawData_test, rawData_train_train, rawData_train_test, creansedData_train_train, creansedData_train_test, creansedData_test, model, result_test, result

    
if __name__ == "__main__":
    #data = main()
    rawData_train, rawData_train_train, rawData_train_test, rawData_test, creansedData_train_train, creansedData_train_test, creansedData_test, model, result_test, result = main()
    
    
    