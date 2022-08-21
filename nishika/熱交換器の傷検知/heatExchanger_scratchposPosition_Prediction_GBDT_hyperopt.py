# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 09:37:59 2022

@author: aokis
"""
import glob
import csv
from concurrent import futures

import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import log_loss

import xgboost as xgb

class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {'objective': 'binary:logistic', 'eval_metric':'auc', 'silent': 1, 'random_state': 71}
        params.update(self.params)
        num_round = 200
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

def data_read(filename):
    data = pd.read_csv(filename)
    data.rename(columns = {"Unnamed: 0": ""}, inplace=True)
    return data.set_index("")


def param_search(label, data):
    def score(params):
        # パラメータを与えたときに最小化する評価指標を指定する
        # 具体的には、モデルにパラメータを指定して学習・予測させた場合のスコアを返すようにする

        # max_depthの型を整数型に修正する
        params['max_depth'] = int(params['max_depth'])

        # Modelクラスを定義しているものとする
        # Modelクラスは、fitで学習し、predictで予測値の確率を出力する
        model = Model(params)
        model.fit(tr_x, tr_y, va_x, va_y)
        va_pred = model.predict(va_x)
        
        fpr, tpr, thresholds = roc_curve(va_y["target"], va_pred)
        score = auc(fpr, tpr)
        print(f'params: {params}, auc: {score:.4f}')
    
        # 情報を記録しておく
        history.append((params, score))

        return {'loss': score, 'status': STATUS_OK}
    
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.1,
        'gamma': 0.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'min_child_weight': 1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 71,
    }
    
    param_space = {
        'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
        'max_depth': hp.quniform('max_depth', 3, 12, 1),
        'subsample': hp.quniform('subsample', 0.4, 0.95, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.95, 0.05),
        'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
        # 余裕があればalpha, lambdaも調整する
        'alpha' : hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
        'lambda' : hp.loguniform('lambda', np.log(1e-6), np.log(10.0)),
    }
    
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    tr_idx, va_idx = list(kf.split(data))[0]
    tr_x, va_x = data.iloc[tr_idx], data.iloc[va_idx]
    tr_y, va_y = label.iloc[tr_idx], label.iloc[va_idx]
    
    max_evals = 10
    trials = Trials()
    history = []
    fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    
    history = sorted(history, key=lambda tpl: tpl[1], reverse=True)
    best = history[0]
    print(f'best params:{best[0]}, score:{best[1]:.4f}')
    
    return history

def Training(data, labels, params):
    def_params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eta': 0.1,
        'gamma': 0.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'min_child_weight': 1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 71,
    }
    def_params.update(params)
    
    
    Y = labels["target"]
    X = data
    
    dtrain = xgb.DMatrix(X, label=Y)
    
    model = xgb.train(def_params, dtrain, 200)
    
    return model

def test(data, model, name, Eval):
    with open("submission_" + name + "_" + Eval + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "target"])
        
        pred = model.predict(xgb.DMatrix(data))
        pred[pred < 0] = 0
        pred[pred > 1] = 1
        for index in range(len(pred)):
            
            w.writerow([data.index[index] , pred[index]])
    
    return

def save_searched_param(history, name, Eval):

    keys = list(history[0][0].keys())
    
    with open("sarchedParams_" +name + "_" + Eval +".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["No", "Eval"] + keys)
        
        for i in range(len(history)):
            
            w.writerow([i, history[i][1]] + [history[i][0][key] for key in keys])
            
def main():
    name = "GBDT_2"
    Eval = "Roc"
    
    print("load label")
    labels = data_read("train_label.csv")
    
    print("load train data")
    creansed_data_train = data_read("train_data.csv")
    
    print("load test data")
    creansed_data_test = data_read("test_data.csv")
    
    print("params search start")
    history = param_search(labels, creansed_data_train)
    
    print("train")
    model = Training(creansed_data_train, labels, history[0][0])
    print("test")
    test(creansed_data_test, model, name, Eval)
    print("histry")
    save_searched_param(history, name, Eval)    
   
    return history


if __name__ == "__main__":
    history = main()
    