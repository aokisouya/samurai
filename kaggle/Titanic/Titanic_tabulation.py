# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:02:29 2022

@author: aokis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def Odds_sort(data):
    
    #index = data["PassengerId"]
    data = data.set_index("PassengerId", verify_integrity = True)
    
    #Age
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_dead = data[data["Survived"] == 0]["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    rate = histgram_Age_Survived/ histgram_Age_dead
    rate = rate.sort_values()
    
    ax_1.bar(rate.keys().astype(str), rate)

    ax_1.set_ylabel("Odds")
    ax_1.set_xlabel("Survivel Odds by Age")
    fig_1.autofmt_xdate(rotation = 35)
    fig_1.savefig(".\\graph_image\\Survival odds sort\\Age.png")
    
    #Pclass
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111)
    
    typeName = sorted(data["Pclass"].unique().astype(str))
    
    histgram_Pclass_Dead = data[data["Survived"] == 0]["Pclass"].value_counts().sort_index()
    histgram_Pclass_Survived = data[data["Survived"] == 1]["Pclass"].value_counts().sort_index()
    
    rate = histgram_Pclass_Survived/ histgram_Pclass_Dead
    rate = rate.sort_values()
    
    ax_2.bar(rate.keys().astype(str), rate)

    ax_2.set_ylabel("Odds")
    ax_2.set_xlabel("Survivel Odds by Pclass")
    fig_2.savefig(".\\graph_image\\Survival odds sort\\Pclass.png")
    
    
    
    #Sex
    #質的変数でhistgramをつ
    fig_3 = plt.figure()
    ax_3 = fig_3.add_subplot(111)
    
    typeName = sorted(data["Sex"].unique().astype(str))
    
    histgram_Sex_dead = data[data["Survived"] == 0]["Sex"].value_counts().sort_index()
    histgram_Sex_Survived = data[data["Survived"] == 1]["Sex"].value_counts().sort_index()
    
    rate = histgram_Sex_Survived/ histgram_Sex_dead
    rate = rate.sort_values()
    
    ax_3.bar(rate.keys().astype(str), rate)

    ax_3.set_ylabel("Odds")
    ax_3.set_xlabel("Survivel Odds by Sex")
    fig_3.savefig(".\\graph_image\\Survival odds sort\\Sex.png")
    
    #SibSp
    fig_4 = plt.figure()
    ax_4 = fig_4.add_subplot(111)
    
    typeName = sorted(data["SibSp"].unique().astype(str))
    
    histgram_SibSp_dead = data[data["Survived"] == 0]["SibSp"].value_counts().sort_index()
    histgram_SibSp_Survived = data[data["Survived"] == 1]["SibSp"].value_counts().sort_index()
    
    rate = histgram_SibSp_Survived/ histgram_SibSp_dead
    rate = rate.sort_values()
    
    ax_4.bar(rate.keys().astype(str), rate)

    ax_4.set_ylabel("Odds")
    ax_4.set_xlabel("Survivel Odds by SibSp")
    fig_4.savefig(".\\graph_image\\Survival odds sort\\SibSp.png")
        
    #Parch
    fig_5 = plt.figure()
    ax_5 = fig_5.add_subplot(111)
    
    typeName = sorted(data["Parch"].unique().astype(str))
    
    histgram_Parch_dead = data[data["Survived"] == 0]["Parch"].value_counts().sort_index()
    histgram_Parch_Survived = data[data["Survived"] == 1]["Parch"].value_counts().sort_index()
    
    rate = histgram_Parch_Survived/ histgram_Parch_dead
    rate = rate.sort_values()
    
    ax_5.bar(rate.keys().astype(str), rate)

    ax_5.set_ylabel("Odds")
    ax_5.set_xlabel("Survivel odds by Parch")
    fig_5.savefig(".\\graph_image\\Survival odds sort\\Parch.png")
    
    #Fare
    fig_6 = plt.figure()
    ax_6 = fig_6.add_subplot(111)
    
    min_max = (data["Fare"].min(),data["Fare"].max())
    
    bins = list(range(0,551, 50))
    
    histgram_Fare_dead = data[data["Survived"] == 0]["Fare"].value_counts(bins = bins).sort_index()
    histgram_Fare_Survived = data[data["Survived"] == 1]["Fare"].value_counts(bins = bins).sort_index()
    
    rate = histgram_Fare_Survived/ histgram_Fare_dead
    rate = rate.sort_values()
    
    ax_6.bar(rate.keys().astype(str), rate)

    ax_6.set_ylabel("Odds")
    ax_6.set_xlabel("Survivel odds by Fare")
    fig_6.autofmt_xdate(rotation = 35)
    fig_6.savefig(".\\graph_image\\Survival odds sort\\Fare.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Cabin"][i]) == str:
            data.loc[i, "Cabin"] =  data["Cabin"][i][0]
        else:
            data.loc[i, "Cabin"] = "N"
    fig_7 = plt.figure()
    ax_7 = fig_7.add_subplot(111)
    
    typeName = sorted(data["Cabin"].unique().astype(str))
    
    histgram_Cabin_dead = data[data["Survived"] == 0]["Cabin"].value_counts().sort_index()
    histgram_Cabin_Survived = data[data["Survived"] == 1]["Cabin"].value_counts().sort_index()
    
    rate = histgram_Cabin_Survived/ histgram_Cabin_dead
    rate = rate.sort_values()
    
    ax_7.bar(rate.keys().astype(str), rate)

    ax_7.set_ylabel("Odds")
    ax_7.set_xlabel("Survivel odds by Cabin")
    fig_7.savefig(".\\graph_image\\Survival odds sort\\Cabin.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Embarked"][i]) == str:
            data.loc[i, "Embarked"] =  data["Embarked"][i][0]
        else:
            data.loc[i, "Embarked"] = "N"
    fig_8 = plt.figure()
    ax_8 = fig_8.add_subplot(111)
    
    typeName = sorted(data["Embarked"].unique().astype(str))
    
    histgram_Embarked_dead = data[data["Survived"] == 0]["Embarked"].value_counts().sort_index()
    histgram_Embarked_Survived = data[data["Survived"] == 1]["Embarked"].value_counts().sort_index()
    
    rate = histgram_Embarked_Survived/ histgram_Embarked_dead
    rate = rate.sort_values()
    
    ax_8.bar(rate.keys().astype(str), rate)

    ax_8.set_ylabel("Odds")
    ax_8.set_xlabel("Survivel odds by Embarked")
    fig_8.savefig(".\\graph_image\\Survival odds sort\\Embarked.png")
    
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
    
    typeName = sorted(data["Ticket"].unique().astype(str))
    
    histgram_Ticket_dead = data[data["Survived"] == 0]["Ticket"].value_counts().sort_index()
    histgram_Ticket_Survived = data[data["Survived"] == 1]["Ticket"].value_counts().sort_index()
    
    rate = histgram_Ticket_Survived/ histgram_Ticket_dead
    rate = rate.sort_values()
    
    ax_9.bar(rate.keys().astype(str), rate)

    ax_9.set_ylabel("Odds")
    ax_9.set_xlabel("Survivel odds by Ticket")
    fig_9.savefig(".\\graph_image\\Survival odds sort\\Ticket.png")
    
    """
    #Age_Sex_Cross
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_Survived = data[data["Survived"] == 0]["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    ax_1.bar(histgram_Age_total.keys().astype(str), histgram_Age_Survived/ histgram_Age_total)

    ax_1.set_ylabel("rate")
    ax_1.set_xlabel("Survivel odds by age")
    
    """
    plt.show()

def SurvivalRate_sort(data):
    
    #index = data["PassengerId"]
    data = data.set_index("PassengerId", verify_integrity = True)
    
    #Age
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_total = data["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    rate = histgram_Age_Survived/ histgram_Age_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_1.bar(rate.keys().astype(str), rate)

    ax_1.set_ylabel("rate")
    ax_1.set_xlabel("Survivel rate by age")
    fig_1.autofmt_xdate(rotation = 35)
    fig_1.savefig(".\\graph_image\\Survival rate sort\\age.png")
    
    #Pclass
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111)
    
    typeName = sorted(data["Pclass"].unique().astype(str))
    
    histgram_Pclass_total = data["Pclass"].value_counts().sort_index()
    histgram_Pclass_Survived = data[data["Survived"] == 1]["Pclass"].value_counts().sort_index()
    
    rate = histgram_Pclass_Survived/ histgram_Pclass_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_2.bar(rate.keys().astype(str), rate)

    ax_2.set_ylabel("rate")
    ax_2.set_xlabel("Survivel rate by Pclass")
    fig_2.savefig(".\\graph_image\\Survival rate sort\\Pclass.png")
    
    
    #Sex
    #質的変数でhistgramをつ
    fig_3 = plt.figure()
    ax_3 = fig_3.add_subplot(111)
    
    typeName = sorted(data["Sex"].unique().astype(str))
    
    histgram_Sex_total = data["Sex"].value_counts().sort_index()
    histgram_Sex_Survived = data[data["Survived"] == 1]["Sex"].value_counts().sort_index()
    
    rate = histgram_Sex_Survived/ histgram_Sex_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_3.bar(rate.keys().astype(str), rate)

    ax_3.set_ylabel("rate")
    ax_3.set_xlabel("Survivel rate by Sex")
    fig_3.savefig(".\\graph_image\\Survival rate sort\\Sex.png")
    
    #SibSp
    fig_4 = plt.figure()
    ax_4 = fig_4.add_subplot(111)
    
    typeName = sorted(data["SibSp"].unique().astype(str))
    
    histgram_SibSp_total = data["SibSp"].value_counts().sort_index()
    histgram_SibSp_Survived = data[data["Survived"] == 1]["SibSp"].value_counts().sort_index()
    
    rate = histgram_SibSp_Survived/ histgram_SibSp_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_4.bar(rate.keys().astype(str), rate)

    ax_4.set_ylabel("rate")
    ax_4.set_xlabel("Survivel rate by SibSp")
    fig_4.savefig(".\\graph_image\\Survival rate sort\\SibSp.png")
        
    #Parch
    fig_5 = plt.figure()
    ax_5 = fig_5.add_subplot(111)
    
    typeName = sorted(data["Parch"].unique().astype(str))
    
    histgram_Parch_total = data["Parch"].value_counts().sort_index()
    histgram_Parch_Survived = data[data["Survived"] == 1]["Parch"].value_counts().sort_index()
    
    rate = histgram_Parch_Survived/ histgram_Parch_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_5.bar(rate.keys().astype(str), rate)

    ax_5.set_ylabel("rate")
    ax_5.set_xlabel("Survivel rate by Parch")
    fig_5.savefig(".\\graph_image\\Survival rate sort\\Parch.png")
    
    #Fare
    fig_6 = plt.figure()
    ax_6 = fig_6.add_subplot(111)
    
    min_max = (data["Fare"].min(),data["Fare"].max())
    
    bins = list(range(0,551, 50))
    
    histgram_Fare_total = data["Fare"].value_counts(bins = bins).sort_index()
    histgram_Fare_Survived = data[data["Survived"] == 1]["Fare"].value_counts(bins = bins).sort_index()
    
    rate = histgram_Fare_Survived/ histgram_Fare_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_6.bar(rate.keys().astype(str), rate)

    ax_6.set_ylabel("rate")
    ax_6.set_xlabel("Survivel rate by Fare")
    fig_6.autofmt_xdate(rotation = 35)
    fig_6.savefig(".\\graph_image\\Survival rate sort\\Fare.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Cabin"][i]) == str:
            data.loc[i, "Cabin"] =  data["Cabin"][i][0]
        else:
            data.loc[i, "Cabin"] = "N"
    fig_7 = plt.figure()
    ax_7 = fig_7.add_subplot(111)
    
    typeName = sorted(data["Cabin"].unique().astype(str))
    
    histgram_Cabin_total = data["Cabin"].value_counts().sort_index()
    histgram_Cabin_Survived = data[data["Survived"] == 1]["Cabin"].value_counts().sort_index()
    
    rate = histgram_Cabin_Survived/ histgram_Cabin_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_7.bar(rate.keys().astype(str), rate)

    ax_7.set_ylabel("rate")
    ax_7.set_xlabel("Survivel rate by Cabin")
    fig_7.savefig(".\\graph_image\\Survival rate sort\\Cabin.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Embarked"][i]) == str:
            data.loc[i, "Embarked"] =  data["Embarked"][i][0]
        else:
            data.loc[i, "Embarked"] = "N"
    fig_8 = plt.figure()
    ax_8 = fig_8.add_subplot(111)
    
    typeName = sorted(data["Embarked"].unique().astype(str))
    
    histgram_Embarked_total = data["Embarked"].value_counts().sort_index()
    histgram_Embarked_Survived = data[data["Survived"] == 1]["Embarked"].value_counts().sort_index()
    
    rate = histgram_Embarked_Survived/ histgram_Embarked_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_8.bar(rate.keys().astype(str), rate)

    ax_8.set_ylabel("rate")
    ax_8.set_xlabel("Survivel rate by Embarked")
    fig_8.savefig(".\\graph_image\\Survival rate sort\\Embarked.png")
    
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
    
    typeName = sorted(data["Ticket"].unique().astype(str))
    
    histgram_Ticket_total = data["Ticket"].value_counts().sort_index()
    histgram_Ticket_Survived = data[data["Survived"] == 1]["Ticket"].value_counts().sort_index()
    
    rate = histgram_Ticket_Survived/ histgram_Ticket_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_9.bar(rate.keys().astype(str), rate)

    ax_9.set_ylabel("rate")
    ax_9.set_xlabel("Survivel rate by Ticket")
    fig_9.savefig(".\\graph_image\\Survival rate\\Ticket.png")

    plt.show()
def Odds(data):
    
    #index = data["PassengerId"]
    data = data.set_index("PassengerId", verify_integrity = True)
    
    #Age
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_dead = data[data["Survived"] == 0]["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    ax_1.bar(histgram_Age_dead.keys().astype(str), histgram_Age_Survived/ histgram_Age_dead)

    ax_1.set_ylabel("Odds")
    ax_1.set_xlabel("Survivel Odds by Age")
    fig_1.autofmt_xdate(rotation = 35)
    fig_1.savefig(".\\graph_image\\Survival odds\\Age.png")
    
    #Pclass
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111)
    
    typeName = sorted(data["Pclass"].unique().astype(str))
    
    histgram_Pclass_Dead = data[data["Survived"] == 0]["Pclass"].value_counts().sort_index()
    histgram_Pclass_Survived = data[data["Survived"] == 1]["Pclass"].value_counts().sort_index()
    
    ax_2.bar(typeName, histgram_Pclass_Survived/ histgram_Pclass_Dead)

    ax_2.set_ylabel("Odds")
    ax_2.set_xlabel("Survivel Odds by Pclass")
    fig_2.savefig(".\\graph_image\\Survival odds\\Pclass.png")
    
    
    
    #Sex
    #質的変数でhistgramをつ
    fig_3 = plt.figure()
    ax_3 = fig_3.add_subplot(111)
    
    typeName = sorted(data["Sex"].unique().astype(str))
    
    histgram_Sex_dead = data[data["Survived"] == 0]["Sex"].value_counts().sort_index()
    histgram_Sex_Survived = data[data["Survived"] == 1]["Sex"].value_counts().sort_index()
    
    ax_3.bar(typeName, histgram_Sex_Survived/ histgram_Sex_dead)

    ax_3.set_ylabel("Odds")
    ax_3.set_xlabel("Survivel Odds by Sex")
    fig_3.savefig(".\\graph_image\\Survival odds\\Sex.png")
    
    #SibSp
    fig_4 = plt.figure()
    ax_4 = fig_4.add_subplot(111)
    
    typeName = sorted(data["SibSp"].unique().astype(str))
    
    histgram_SibSp_dead = data[data["Survived"] == 0]["SibSp"].value_counts().sort_index()
    histgram_SibSp_Survived = data[data["Survived"] == 1]["SibSp"].value_counts().sort_index()
    
    ax_4.bar(histgram_SibSp_dead.keys().astype(str), histgram_SibSp_Survived/ histgram_SibSp_dead)

    ax_4.set_ylabel("rate")
    ax_4.set_xlabel("Survivel Odds by SibSp")
    fig_4.savefig(".\\graph_image\\Survival odds\\SibSp.png")
        
    #Parch
    fig_5 = plt.figure()
    ax_5 = fig_5.add_subplot(111)
    
    typeName = sorted(data["Parch"].unique().astype(str))
    
    histgram_Parch_dead = data[data["Survived"] == 0]["Parch"].value_counts().sort_index()
    histgram_Parch_Survived = data[data["Survived"] == 1]["Parch"].value_counts().sort_index()
    
    ax_5.bar(histgram_Parch_dead.keys().astype(str), histgram_Parch_Survived/ histgram_Parch_dead)

    ax_5.set_ylabel("Odds")
    ax_5.set_xlabel("Survivel odds by Parch")
    fig_5.savefig(".\\graph_image\\Survival odds\\Parch.png")
    
    #Fare
    fig_6 = plt.figure()
    ax_6 = fig_6.add_subplot(111)
    
    min_max = (data["Fare"].min(),data["Fare"].max())
    
    bins = list(range(0,551, 50))
    
    histgram_Fare_dead = data[data["Survived"] == 0]["Fare"].value_counts(bins = bins).sort_index()
    histgram_Fare_Survived = data[data["Survived"] == 1]["Fare"].value_counts(bins = bins).sort_index()
    
    ax_6.bar(histgram_Fare_dead.keys().astype(str), histgram_Fare_Survived/ histgram_Fare_dead, tick_label = histgram_Fare_dead.keys().astype(str))

    ax_6.set_ylabel("Odds")
    ax_6.set_xlabel("Survivel odds by Fare")
    fig_6.autofmt_xdate(rotation = 35)
    fig_6.savefig(".\\graph_image\\Survival odds\\Fare.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Cabin"][i]) == str:
            data.loc[i, "Cabin"] =  data["Cabin"][i][0]
        else:
            data.loc[i, "Cabin"] = "N"
    fig_7 = plt.figure()
    ax_7 = fig_7.add_subplot(111)
    
    typeName = sorted(data["Cabin"].unique().astype(str))
    
    histgram_Cabin_dead = data[data["Survived"] == 0]["Cabin"].value_counts().sort_index()
    histgram_Cabin_Survived = data[data["Survived"] == 1]["Cabin"].value_counts().sort_index()
    
    ax_7.bar(histgram_Cabin_dead.keys().astype(str), histgram_Cabin_Survived/ histgram_Cabin_dead)

    ax_7.set_ylabel("Odds")
    ax_7.set_xlabel("Survivel odds by Cabin")
    fig_7.savefig(".\\graph_image\\Survival odds\\Cabin.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Embarked"][i]) == str:
            data.loc[i, "Embarked"] =  data["Embarked"][i][0]
        else:
            data.loc[i, "Embarked"] = "N"
    fig_8 = plt.figure()
    ax_8 = fig_8.add_subplot(111)
    
    typeName = sorted(data["Embarked"].unique().astype(str))
    
    histgram_Embarked_dead = data[data["Survived"] == 0]["Embarked"].value_counts().sort_index()
    histgram_Embarked_Survived = data[data["Survived"] == 1]["Embarked"].value_counts().sort_index()
    
    ax_8.bar(typeName, histgram_Embarked_Survived/ histgram_Embarked_dead)

    ax_8.set_ylabel("Odds")
    ax_8.set_xlabel("Survivel odds by Embarked")
    fig_8.savefig(".\\graph_image\\Survival odds\\Embarked.png")
    
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
    
    typeName = sorted(data["Ticket"].unique().astype(str))
    
    histgram_Ticket_dead = data[data["Survived"] == 0]["Ticket"].value_counts().sort_index()
    histgram_Ticket_Survived = data[data["Survived"] == 1]["Ticket"].value_counts().sort_index()
    
    ax_9.bar(histgram_Ticket_dead.keys().astype(str), histgram_Ticket_Survived/ histgram_Ticket_dead)

    ax_9.set_ylabel("Odds")
    ax_9.set_xlabel("Survivel odds by Ticket")
    fig_9.savefig(".\\graph_image\\Survival odds\\Ticket.png")
    
    """
    #Age_Sex_Cross
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_Survived = data[data["Survived"] == 0]["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    ax_1.bar(histgram_Age_total.keys().astype(str), histgram_Age_Survived/ histgram_Age_total)

    ax_1.set_ylabel("rate")
    ax_1.set_xlabel("Survivel odds by age")
    
    """
    plt.show()

def SurvivalRate(data):
    
    #index = data["PassengerId"]
    data = data.set_index("PassengerId", verify_integrity = True)
    
    #Age
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_total = data["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    ax_1.bar(histgram_Age_total.keys().astype(str), histgram_Age_Survived/ histgram_Age_total)

    ax_1.set_ylabel("rate")
    ax_1.set_xlabel("Survivel rate by age")
    fig_1.autofmt_xdate(rotation = 35)
    fig_1.savefig(".\\graph_image\\Survival rate\\age.png")
    
    #Pclass
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111)
    
    typeName = sorted(data["Pclass"].unique().astype(str))
    
    histgram_Pclass_total = data["Pclass"].value_counts().sort_index()
    histgram_Pclass_Survived = data[data["Survived"] == 1]["Pclass"].value_counts().sort_index()
    
    ax_2.bar(typeName, histgram_Pclass_Survived/ histgram_Pclass_total)

    ax_2.set_ylabel("rate")
    ax_2.set_xlabel("Survivel rate by Pclass")
    fig_2.savefig(".\\graph_image\\Survival rate\\Pclass.png")
    
    
    #Sex
    #質的変数でhistgramをつ
    fig_3 = plt.figure()
    ax_3 = fig_3.add_subplot(111)
    
    typeName = sorted(data["Sex"].unique().astype(str))
    
    histgram_Sex_total = data["Sex"].value_counts().sort_index()
    histgram_Sex_Survived = data[data["Survived"] == 1]["Sex"].value_counts().sort_index()
    
    ax_3.bar(typeName, histgram_Sex_Survived/ histgram_Sex_total)

    ax_3.set_ylabel("rate")
    ax_3.set_xlabel("Survivel rate by Sex")
    fig_3.savefig(".\\graph_image\\Survival rate\\Sex.png")
    
    #SibSp
    fig_4 = plt.figure()
    ax_4 = fig_4.add_subplot(111)
    
    typeName = sorted(data["SibSp"].unique().astype(str))
    
    histgram_SibSp_total = data["SibSp"].value_counts().sort_index()
    histgram_SibSp_Survived = data[data["Survived"] == 1]["SibSp"].value_counts().sort_index()
    
    ax_4.bar(histgram_SibSp_total.keys().astype(str), histgram_SibSp_Survived/ histgram_SibSp_total)

    ax_4.set_ylabel("rate")
    ax_4.set_xlabel("Survivel rate by SibSp")
    fig_4.savefig(".\\graph_image\\Survival rate\\SibSp.png")
        
    #Parch
    fig_5 = plt.figure()
    ax_5 = fig_5.add_subplot(111)
    
    typeName = sorted(data["Parch"].unique().astype(str))
    
    histgram_Parch_total = data["Parch"].value_counts().sort_index()
    histgram_Parch_Survived = data[data["Survived"] == 1]["Parch"].value_counts().sort_index()
    
    ax_5.bar(histgram_Parch_total.keys().astype(str), histgram_Parch_Survived/ histgram_Parch_total)

    ax_5.set_ylabel("rate")
    ax_5.set_xlabel("Survivel rate by Parch")
    fig_5.savefig(".\\graph_image\\Survival rate\\Parch.png")
    
    #Fare
    fig_6 = plt.figure()
    ax_6 = fig_6.add_subplot(111)
    
    min_max = (data["Fare"].min(),data["Fare"].max())
    
    bins = list(range(0,551, 50))
    
    histgram_Fare_total = data["Fare"].value_counts(bins = bins).sort_index()
    histgram_Fare_Survived = data[data["Survived"] == 1]["Fare"].value_counts(bins = bins).sort_index()
    
    ax_6.bar(histgram_Fare_total.keys().astype(str), histgram_Fare_Survived/ histgram_Fare_total)

    ax_6.set_ylabel("rate")
    ax_6.set_xlabel("Survivel rate by Fare")
    fig_6.autofmt_xdate(rotation = 35)
    fig_6.savefig(".\\graph_image\\Survival rate\\Fare.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Cabin"][i]) == str:
            data.loc[i, "Cabin"] =  data["Cabin"][i][0]
        else:
            data.loc[i, "Cabin"] = "N"
    fig_7 = plt.figure()
    ax_7 = fig_7.add_subplot(111)
    
    typeName = sorted(data["Cabin"].unique().astype(str))
    
    histgram_Cabin_total = data["Cabin"].value_counts().sort_index()
    histgram_Cabin_Survived = data[data["Survived"] == 1]["Cabin"].value_counts().sort_index()
    
    ax_7.bar(histgram_Cabin_total.keys().astype(str), histgram_Cabin_Survived/ histgram_Cabin_total)

    ax_7.set_ylabel("rate")
    ax_7.set_xlabel("Survivel rate by Cabin")
    fig_7.savefig(".\\graph_image\\Survival rate\\Cabin.png")
    
    #Cabin
    for i in data.index:
        #Cabinをグループごとに変換（客室番号の頭文字と定義）＆nanをNと定義
        if type(data["Embarked"][i]) == str:
            data.loc[i, "Embarked"] =  data["Embarked"][i][0]
        else:
            data.loc[i, "Embarked"] = "N"
    fig_8 = plt.figure()
    ax_8 = fig_8.add_subplot(111)
    
    typeName = sorted(data["Embarked"].unique().astype(str))
    
    histgram_Embarked_total = data["Embarked"].value_counts().sort_index()
    histgram_Embarked_Survived = data[data["Survived"] == 1]["Embarked"].value_counts().sort_index()
    
    ax_8.bar(histgram_Embarked_total.keys().astype(str), histgram_Embarked_Survived/ histgram_Embarked_total)

    ax_8.set_ylabel("rate")
    ax_8.set_xlabel("Survivel rate by Embarked")
    fig_8.savefig(".\\graph_image\\Survival rate\\Embarked.png")
    
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
    
    typeName = sorted(data["Ticket"].unique().astype(str))
    
    histgram_Ticket_total = data["Ticket"].value_counts().sort_index()
    histgram_Ticket_Survived = data[data["Survived"] == 1]["Ticket"].value_counts().sort_index()
    
    ax_9.bar(histgram_Ticket_total.keys().astype(str), histgram_Ticket_Survived/ histgram_Ticket_total)

    ax_9.set_ylabel("rate")
    ax_9.set_xlabel("Survivel rate by Ticket")
    fig_9.savefig(".\\graph_image\\Survival rate\\Ticket.png")
    
    """
    #Age_Sex_Cross
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    bins = list(range(0,85, 5))
    
    histgram_Age_total = data["Age"].value_counts(bins = bins).sort_index()
    histgram_Age_Survived = data[data["Survived"] == 1]["Age"].value_counts(bins = bins).sort_index()
    
    ax_1.bar(histgram_Age_total.keys().astype(str), histgram_Age_Survived/ histgram_Age_total)

    ax_1.set_ylabel("rate")
    ax_1.set_xlabel("Survivel rate by age")
    
    """
    plt.show()
    
def Sex_Age(data):
    
    mode = data["Age"].mode()[0]
    
    data["Age"] = round(data["Age"] // 15) * 15
    
    data["Sex"] = data["Sex"] + "-" + data["Age"].astype(str)
    
    
    histgram_Sex_Age_total = data["Sex"].value_counts().sort_index()
    histgram_Sex_Age_dead = data[data["Survived"] == 0]["Sex"].value_counts().sort_index()
    histgram_Sex_Age_Survived = data[data["Survived"] == 1]["Sex"].value_counts().sort_index()
    
    #Age
    fig_1 = plt.figure()
    ax_1 = fig_1.add_subplot(111)
    
    rate = histgram_Sex_Age_Survived/ histgram_Sex_Age_total
    rate = rate.fillna(0)
    rate = rate.sort_values()
    
    ax_1.bar(rate.keys().astype(str), rate)

    ax_1.set_ylabel("rate")
    ax_1.set_xlabel("Survivel rate by Age")
    fig_1.autofmt_xdate(rotation = 35)
    fig_1.savefig(".\\graph_image\\Sex-Age\\rate.png")
    
    
    fig_2 = plt.figure()
    ax_2 = fig_2.add_subplot(111)
    
    odds = histgram_Sex_Age_Survived/ histgram_Sex_Age_dead
    odds = odds.sort_values()
    
    ax_2.bar(odds.keys().astype(str), odds)
    
    ax_2.set_ylabel("Odds")
    ax_2.set_xlabel("Survivel Odds by Age")
    fig_2.autofmt_xdate(rotation = 35)
    fig_2.savefig(".\\graph_image\\Sex-Age\\odds.png")
    
    return
    
    
    

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

def main():
    
    rawData_train, rawData_test = load_data()
    #SurvivalRate(rawData_train) #グラフ表示
    #Odds(rawData_train)
    #SurvivalRate_sort(rawData_train) #グラフ表示
    #Odds_sort(rawData_train)
    Sex_Age(rawData_train)
    
    return rawData_train
    
if __name__ == "__main__":
    data = main()