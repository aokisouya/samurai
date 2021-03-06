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

def tabulation(data):
    
    #index = data["PassengerId"]
    data = data.set_index("PassengerId", verify_integrity = True)
    
    #Age
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    ax_1 = data[data["Survived"] == 0]["Age"].hist(bins = bins, alpha = 0.5, label = "0:dead", density=True)
    ax_1 = data[data["Survived"] == 1]["Age"].hist(bins = bins, alpha = 0.5, label = "1:Survived", density=True)
    ax_1.legend()
    ax_1.set_ylabel("number of people")
    ax_1.set_xlabel("Age")
    
    #Pclass
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111)
    
    min_max = (data["Pclass"].min(),data["Pclass"].max())
    
    ax_2 = data[data["Survived"] == 0]["Pclass"].hist(bins = 3, alpha = 0.5, label = "0:dead", density=True)
    ax_2 = data[data["Survived"] == 1]["Pclass"].hist(bins = 3, alpha = 0.5, label = "1:Survived", density=True)
    ax_2.legend()
    ax_2.set_xticks([1,2,3])
    ax_2.set_ylabel("number of people")
    ax_2.set_xlabel("Pclass")
    
    #Sex
    #質的変数でhistgramをつ
    fig_3 = plt.figure()
    ax_3 = fig_3.add_subplot(111)
    
    dead = data[data["Survived"] == 0]["Sex"].value_counts()/data[data["Survived"] == 0]["Sex"].count()
    Survived = data[data["Survived"] == 1]["Sex"].value_counts()/data[data["Survived"] == 1]["Sex"].count()
    
    ax_3.bar(dead.index, dead, alpha = 0.5, color = "C0", label = "0:dead")
    ax_3.bar(Survived.index, Survived, alpha = 0.5, color = "C1", label = "1:Survived")
    ax_3.legend()
    ax_3.set_ylabel("number of people")
    ax_3.set_xlabel("Sex")
    
    #SibSp
    fig_4 = plt.figure()
    ax_4 = fig_4.add_subplot(111)
    
    min_max = (data["SibSp"].min(),data["SibSp"].max())
    bins = list(range(min_max[0],min_max[1]))
    
    ax_4 = data[data["Survived"] == 0]["SibSp"].hist(bins = bins, alpha = 0.5, label = "0:dead", range = min_max, density=True)
    ax_4 = data[data["Survived"] == 1]["SibSp"].hist(bins = bins, alpha = 0.5, label = "1:Survived", range = min_max, density=True)
    ax_4.legend()
    ax_4.set_ylabel("number of people")
    ax_4.set_xlabel("SibSp")
        
    #SibSp
    fig_5 = plt.figure()
    ax_5 = fig_5.add_subplot(111)
    
    min_max = (data["Parch"].min(),data["Parch"].max())
    bins = list(range(min_max[0],min_max[1]))
    
    ax_5 = data[data["Survived"] == 0]["Parch"].hist(bins = bins, alpha = 0.5, label = "0:dead", range = min_max, density=True)
    ax_5 = data[data["Survived"] == 1]["Parch"].hist(bins = bins, alpha = 0.5, label = "1:Survived", range = min_max, density=True)
    ax_5.legend()
    ax_5.set_ylabel("number of people")
    ax_5.set_xlabel("Parch")
    
    #Fare
    fig_6 = plt.figure()
    ax_6 = fig_6.add_subplot(111)
    
    min_max = (data["Fare"].min(),data["Fare"].max())
    
    ax_6 = data[data["Survived"] == 0]["Fare"].hist(alpha = 0.5, label = "0:dead", range = min_max, density=True)
    ax_6 = data[data["Survived"] == 1]["Fare"].hist(alpha = 0.5, label = "1:Survived", range = min_max, density=True)
    ax_6.legend()
    ax_6.set_ylabel("number of people")
    ax_6.set_xlabel("Fare")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Cabin"][i]) == str:
            data.loc[i, "Cabin"] =  data["Cabin"][i][0]
        else:
            data.loc[i, "Cabin"] = "N"
    fig_7 = plt.figure()
    ax_7 = fig_7.add_subplot(111)
    
    dead = data[data["Survived"] == 0]["Cabin"].value_counts()/data[data["Survived"] == 0]["Cabin"].count()
    Survived = data[data["Survived"] == 1]["Cabin"].value_counts()/data[data["Survived"] == 1]["Cabin"].count()

    ax_7.bar(dead.index, dead, alpha = 0.5, color = "C0", label = "0:dead")
    ax_7.bar(Survived.index, Survived, alpha = 0.5, color = "C1", label = "1:Survived")
    ax_7.legend()
    ax_7.set_ylabel("number of people")
    ax_7.set_xlabel("Cabin")

    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Embarked"][i]) == str:
            data.loc[i, "Embarked"] =  data["Embarked"][i][0]
        else:
            data.loc[i, "Embarked"] = "N"
    fig_8 = plt.figure()
    ax_8 = fig_8.add_subplot(111)
    
    dead = data[data["Survived"] == 0]["Embarked"].value_counts()/data[data["Survived"] == 0]["Embarked"].count()
    Survived = data[data["Survived"] == 1]["Embarked"].value_counts()/data[data["Survived"] == 1]["Embarked"].count()

    ax_8.bar(dead.index, dead, alpha = 0.5, color = "C0", label = "0:dead")
    ax_8.bar(Survived.index, Survived, alpha = 0.5, color = "C1", label = "1:Survived")
    ax_8.legend()
    ax_8.set_ylabel("number of people")
    ax_8.set_xlabel("Embarked")
    
    #Ticket
    #チケットについて修正
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if data["Ticket"][i][0] in [str(i) for i in range(10)]:
            data.loc[i, "Ticket"] = "number"
        else:
            data.loc[i, "Ticket"] = data["Ticket"][i][0]
    fig_9 = plt.figure()
    ax_9 = fig_9.add_subplot(111)
    
    dead = data[data["Survived"] == 0]["Ticket"].value_counts()/data[data["Survived"] == 0]["Ticket"].count()
    Survived = data[data["Survived"] == 1]["Ticket"].value_counts()/data[data["Survived"] == 1]["Ticket"].count()

    ax_9.bar(dead.index, dead, alpha = 0.5, color = "C0", label = "0:dead")
    ax_9.bar(Survived.index, Survived, alpha = 0.5, color = "C1", label = "1:Survived")
    ax_9.legend()
    ax_9.set_ylabel("number of people")
    ax_9.set_xlabel("Ticket")
    
            
            

    
    plt.show()
    

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
    PassengerId　上客ID 数値（index）
        管理番号であり、生存に対して寄与しない情報の為削除
    Survived 生存判定 1 or 0
        目的変数
    Pclass チケットクラス 1-3
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
    
    #チケットについて修正
    for i in creansedData.index:
        #Ticketを数字だけとそれ以外のチケットでカテゴリ変数に変換
        if creansedData["Ticket"][i][0] in [str(i) for i in range(10)]:
            creansedData.loc[i, "Ticket"] = "number"
        else:
            creansedData.loc[i, "Ticket"] = "Other than number"
            
    mode = creansedData["Age"].mode()[0]
    
    for i in creansedData.index:
        #年齢を15歳区切りでカテゴリ変数へ変換
        if np.isnan(creansedData.loc[i, "Age"]):
            creansedData.loc[i, "Age"] = "U" + str(int(mode//15 * 15))
        else:
            creansedData.loc[i, "Age"] = "U" + str(int(creansedData["Age"][i]//15 * 15))
    
    for i in creansedData.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        #->nanを客室が割り当たっていない人と定義して、客室が割り当たっている人、いない人に修正
        if type(creansedData["Cabin"][i]) == str:
            creansedData.loc[i, "Cabin"] =  "Have"
        else:
            creansedData.loc[i, "Cabin"] = "not have"
            
    for i in creansedData.index:
        #SibSpを1，2，3以上でカテゴリ変数へ変換
        if creansedData["SibSp"][i] >= 3:
            creansedData.loc[i, "SibSp"] = 3
    
    for i in creansedData.index:
        #Parchを1，2，3以上でカテゴリ変数へ変換
        if creansedData["Parch"][i] >= 3:
            creansedData.loc[i, "Parch"] = 3
    
    #欠損のあるデータを削除
    #creansedData = creansedData.dropna(how = "any", axis = 0)           
    
    #Undersampling 
    if "Survived" in creansedData.keys():
        creansedData = Undersampling(creansedData)
    
    #カテゴリ変数をダミー変数に変換
    creansedData = pd.get_dummies(creansedData, columns=["Sex", "Age","SibSp","Parch","Ticket", "Cabin", "Embarked"], drop_first = True)
    
    #(修正箇所)欠損のあるデータを中央値で補完
    #!!!最頻値で補完に修正する
    #ダミー変数も一緒にやっていいかはよく分かっていない　簡単に検索したけど見つからなかった
    #中央値ならダミーにも0/1以外入る事は稀のはずなのでみためも気持ち悪さもあまりない
    creansedData = creansedData.fillna(creansedData.median())
    
    #量的変数を標準化
    for Variable in ["Pclass","Fare"]:
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

def test(model, data, rawData, keys, resultFileName):
    """
    メモ        
    テストデータにCabinのTないやんけ
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
    #tabulation(rawData_train) #グラフ表示
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
    
    
    