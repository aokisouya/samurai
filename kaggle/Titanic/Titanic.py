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
    PassengerId　上客ID 数値（index）
        管理番号であり、生存に対して寄与しない情報の為削除
    Survived 生存判定 1 or 0
        目的変数
    Pclass チケットクラス 0-3
        数値のまま使用    
    Name　名前 char
       　おそらく生存に対して寄与しない情報の為削除
    Sex　性別別 male or female
        ダミー変数へ変換
    Age　年齢　数値
        数値のまま使用
        nullは欠損として行削除
        年齢を量的データとして扱っていいモノ？
        赤子、少年、青年、成人、老人　のように年齢で区切ってカテゴリとして扱ったほうが実情に近そう。
        運動性能的な面だけで見ても20歳前後がピークで、年齢にたいして比例関係ではなさそうなので。
        一旦、量的データのままで作成
    SibSp　タイタニック号に乗っている兄弟/配偶者の数　数値
        数値のまま使用
    Parch　タイタニック号に乗っている親/子供の数　数値
        数値のまま使用
    Ticket　”char 数値” or "数値"
        計681種類。
        同じチケットを持っている人もいる
        扱い困ったので一旦削除
    Fare　運賃（単位不明）
        数値のまま使用
    Cabin 客室番号？　 77.1%値がnull。　大部屋雑魚寝多数とか船員がnullとしてはいっている？
        計148種類。
        ただ、なにがしか因果関係が見られないとは言いきれないのでどうにかして使いたい。
        数値の前のアルファベットが船内のおおよその位置を表しているのであれば、
        'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T' or null
        の9種類なので扱いやすそう？        
        
    Embarked　乗船した港　'S', 'C', 'Q'　or null
        nullは欠損として行削除

    メモ
    チケットで数値のみとそれ以外でデータに特徴が無いか
    ticketA = pd.DataFrame([rawdata.loc[i] for i in range(rawdata.shape[0]) if rawdata.loc[i]["Ticket"][0] in  [str(i) for i in range(10)]])
    ticketB = pd.DataFrame([rawdata.loc[i] for i in range(rawdata.shape[0]) if rawdata.loc[i]["Ticket"][0] not in [str(i) for i in range(10)]])

    """
    #PassengerIdをIndexに変換
    creansedData = rawData.set_index("PassengerId", verify_integrity = True)
    
    #Name削除, 一旦Ticketも削除
    creansedData = creansedData.drop(columns = "Name", axis=1)
    creansedData = creansedData.drop(columns = "Ticket", axis=1)
    
    for i in creansedData.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(creansedData["Cabin"][i]) == str:
            creansedData.loc[i, "Cabin"] =  creansedData["Cabin"][i][0]
        else:
            creansedData.loc[i, "Cabin"] = "N"
    
    #欠損のあるデータを削除
    #creansedData = creansedData.dropna(how = "any", axis = 0)           
    
    #Undersampling 
    if "Survived" in creansedData.keys():
        creansedData = Undersampling(creansedData)
    
    #カテゴリ変数をダミー変数に変換
    creansedData = pd.get_dummies(creansedData, columns=["Sex", "Cabin", "Embarked"], drop_first = True)
    
    #!!!(修正箇所)欠損のあるデータを中央値で補完
    #ダミー変数も一緒にやっていいかはよく分かっていない　簡単に検索したけど見つからなかった
    #中央値ならダミーにも0/1以外入る事は稀のはずなのでみためも気持ち悪さもあまりない
    creansedData = creansedData.fillna(creansedData.median())
            

    #量的変数を標準化
    for Variable in ["Pclass","Age","SibSp","Parch","Fare"]:
        creansedData[Variable] = preprocessing.minmax_scale(creansedData[Variable])
    
    return creansedData

def Training(data):
    Y = data["Survived"]
    X = data[[Variable for Variable in data.keys() if Variable != "Survived"]]
    
    model = LogisticRegression(C = 1, random_state = 0) 
    model.fit(X, Y)
    
    train_score = format(model.score(X, Y))
    print('正解率(train):', train_score)
    
    return model

def test(model, data, rawData):
    """
    メモ        
    テストデータにCabinのTないやんけ
    """
    
    #ダミー変数が足りていなかったので足しています。
    data.insert(loc = 12, column = "Cabin_T", value = 0)
    
    #bias = model.intercept_
    #weight = model.coef_
    

    result = model.predict(data)
    
    
    with open("./result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PassengerId","Survived"])
        for i in range(len(result)):
            w.writerow([rawData["PassengerId"][i], result[i]])
        
    
    return result
    
def main():
    
    result = None
    rawData_train, rawData_test = load_data()
    creansedData_train = data_creansing(rawData_train)
    creansedData_test = data_creansing(rawData_test)
    model = Training(creansedData_train)
    
    
    result = test(model, creansedData_test, rawData_test)
    return rawData_train, rawData_test, creansedData_train, creansedData_test, model, result
    
    
    
    
        
        
if __name__ == "__main__":
    rawData_train, rawData_test, creansedData_train, creansedData_test, model, result = main()
    
    
    