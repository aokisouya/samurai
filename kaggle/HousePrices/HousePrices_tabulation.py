# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:02:29 2022

@author: aokis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import seaborn as sns
sns.set()
plt.rcParams['font.family'] = "MS Gothic"

import csv

def load_dataDescription():
    return pd.read_csv(r".\変数整理.csv")
    
def plot_graph(data):
    dataDescription_list = load_dataDescription()
    
    other_categoryVariable = ["BedroomAbvGr"]
    dataDescription_dic = {}
    
    for key in data.keys():
        if key in ["SalePrice" ,'Id']:
            """
            目的変数外はグラフにする必要がないので、描画を行わない。
            """
            continue
        elif data[key].dtype == object or key in other_categoryVariable:
            """
            カテゴリ変数はカテゴリ毎の箱ひげ図を図示
            """
            continue
            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.boxplot([data["SalePrice"][data[key] == category] for category in data[key].unique()])
            ax.set_xticklabels(data[key].unique())
            ax.set_title(dataDescription_list["Description(jp)"][dataDescription_list["Variable Name"] == key].iloc[-1])
            ax.set_xlabel(key)
            ax.set_ylabel("SalePrice")
            fig.autofmt_xdate(rotation = 35)
            fig.savefig(".\\graph_image3\\" + key + ".png")
            """
            key = "YrSold"
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.boxplot([data["SalePrice"][data[key] == category] for category in sorted(data[key].unique())])
            ax.set_xticklabels(sorted(data[key].unique()))
            ax.set_title(dataDescription_list["Description(jp)"][dataDescription_list["Variable Name"] == key].iloc[-1])
            ax.set_xlabel(key)
            ax.set_ylabel("SalePrice")
            fig.autofmt_xdate(rotation = 35)
            plt.show()
            
            
            fig.savefig(".\\graph_image3\\" + key + ".png")
            """
        else:
            """
            量的変数はカテゴリ毎の散布図を図示
            """
            print(key)
            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.scatter(data[key], data["SalePrice"], alpha = 0.2)
            ax.set_title(dataDescription_list["Description(jp)"][dataDescription_list["Variable Name"] == key].iloc[-1])
            ax.set_xlabel(key)
            ax.set_ylabel("SalePrice")
            fig.savefig(".\\graph_image3\\" + key + ".png")
            """
            key = "BsmtUnfSF"
            
            fig = plt.figure()
            round_value = 200
            data[key] = round(data[key]/round_value) * round_value
            ax = fig.add_subplot(111)
            ax.boxplot([data["SalePrice"][data[key] == category] for category in sorted(data[key].unique())])
            ax.set_xticklabels([x if np.isnan(x) else int(x) for x in sorted(data[key].unique())])
            ax.set_title(dataDescription_list["Description(jp)"][dataDescription_list["Variable Name"] == key].iloc[-1])
            ax.set_xlabel(key)
            ax.set_ylabel("SalePrice")
            fig.autofmt_xdate(rotation = 35)
            plt.show()
            
            fig.savefig(".\\graph_image3\\" + key + ".png")
            
            """
def plot_graph_2(data):
    dataDescription_list = load_dataDescription()
    
    other_categoryVariable = ["2ndFlrSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","BedroomAbvGr", "BsmtFullBath", 
                              "BsmtHalfBath", "Fireplaces", "FullBath", "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold",
                              "MSSubClass", "OverallCond", "TotRmsAbvGrd", "YrSold"]
    dataDescription_dic = {}
    
    for key in data.keys():
        if key in ["SalePrice"]:
            """
            目的変数外はグラフにする必要がないので、描画を行わない。
            """
            continue
        
        elif data[key].dtype == object or key in other_categoryVariable:
            """
            カテゴリ変数はカテゴリ毎の箱ひげ図を図示
            """
            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.boxplot([data["SalePrice"][data[key] == category] for category in data[key].unique()])
            ax.set_xticklabels(data[key].unique())
            ax.set_title(dataDescription_list["Description(jp)"][dataDescription_list["Variable Name"] == key].iloc[-1])
            ax.set_xlabel(key)
            ax.set_ylabel("SalePrice")
            fig.autofmt_xdate(rotation = 35)
            fig.savefig(".\\graph_image2\\" + key + ".png")
        else:
            """
            量的変数はカテゴリ毎の散布図を図示
            """
            fig = plt.Figure()
            ax = fig.add_subplot(111)
            ax.scatter(data[key], data["SalePrice"], alpha = 0.2)
            ax.set_title(dataDescription_list["Description(jp)"][dataDescription_list["Variable Name"] == key].iloc[-1])
            ax.set_xlabel(key)
            ax.set_ylabel("SalePrice")
            fig.savefig(".\\graph_image2\\" + key + ".png")
            
def load_data():
    """
    Returns
    -------
    TYPE
        DataFrame

    """
    rawData_train = pd.read_csv(r".\data\train.csv")
    rawData_test = pd.read_csv(r".\data\test.csv")
    return rawData_train, rawData_test

def data_describe(data):
    keys = [ "SalePrice","MSSubClass","MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition"]
    data_describe = {}
    with open("data_describe.csv", "w", newline = "") as F:
        w = csv.writer(F, delimiter = ",")
        
        w.writerow(["keyname","mean","std","min","25%","50%","75%", "max"])
        
        for key in keys:
            if data[key].dtype == object:
                data_describe[key] = str(data[key].unique())
                w.writerow([key, data_describe[key].replace("\n", " ")])
            else:
                data_describe[key] = data[key].describe()
                w.writerow([key]+list(data_describe[key])[1:])
            
    return 
    
def data_creanse(data):
    
    
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
    
    return data

def correlation_coefficient(data):
        
    categoryVariable = [key for key in data if data[key].dtype == object]
    other_categoryVariable = ["2ndFlrSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","BedroomAbvGr", "BsmtFullBath", 
                              "BsmtHalfBath", "Fireplaces", "FullBath", "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold",
                              "MSSubClass", "OverallCond", "TotRmsAbvGrd", "YrSold"]
    
    #カテゴリ変数をダミー変数に変換
    data = pd.get_dummies(data, columns=categoryVariable + other_categoryVariable, drop_first = True)
    
    corr = data.corr()
    
    with open("corr.csv", "w", newline = "") as F:
        w = csv.writer(F, delimiter = ",")
        
        w.writerow([""] + list(corr.columns.values))
        for key in corr.columns.values:
            w.writerow([key] + list(corr[key].values))
        
def vif(data):
    categoryVariable = [key for key in data if data[key].dtype == object]
    other_categoryVariable = ["2ndFlrSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","BedroomAbvGr", "BsmtFullBath", 
                              "BsmtHalfBath", "Fireplaces", "FullBath", "GarageCars", "HalfBath", "KitchenAbvGr", "MoSold",
                              "MSSubClass", "OverallCond", "TotRmsAbvGrd", "YrSold"]
    
    #カテゴリ変数をダミー変数に変換
    data = pd.get_dummies(data, columns=categoryVariable + other_categoryVariable, drop_first = True)
    
    X = data.drop(columns = ["SalePrice"] , axis=1)
    y = data["SalePrice"]
    
    with open("vif", "w", newline = "") as F:
        w = csv.writer(F, delimiter = ",")
        
        w.writerow([""] + list(corr.columns.values))
        for key in corr.columns.values:
            w.writerow([key] + list(corr[key].values))
    
    

def main():
    
    rawData_train, rawData_test = load_data()

    #data_describe(rawData_train)
    #plot_graph(rawData_train)
    
    CreandedData_train = data_creanse(rawData_train)
    #correlation_coefficient(CreandedData_train)
    vif(CreandedData_train)
    #plot_graph_2(CreandedData_train)
    
    
    return rawData_train
    
if __name__ == "__main__":
    data = main()