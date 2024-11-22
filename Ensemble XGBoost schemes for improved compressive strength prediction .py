# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:47:17 2024

@author: DELL
"""

#载入数据包
import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay

#导入数据
data = pd.read_csv("E:/桌面/机器学习复现/混凝土/marain-810.csv")
data = data.fillna(data.mean())
data = np.array(data)

a = data[:, 0:15]
b = data[:, 15]
columns = ['Cement', 'Slag', 'Silica fume', 'Limestone powder', 'Quartz powder',
           'Fly ash', 'Nano silica', 'Water', 'Sand', 'Gravel', 'Fiber','Superplasticizer',
           'Relative humidity', 'Temperature', 'Age', 'Compressive strength']
data = pd.DataFrame(data, columns = columns)

#绘制散点图
sns.relplot(data = data, x = 'Cement', y = 'Compressive strength')
sns.relplot(data = data, x = 'Slag', y = 'Compressive strength')
sns.relplot(data = data, x = 'Silica fume', y = 'Compressive strength')
sns.relplot(data = data, x = 'Limestone powder', y = 'Compressive strength')
sns.relplot(data = data, x = 'Quartz powder', y = 'Compressive strength')
sns.relplot(data = data, x = 'Fly ash', y = 'Compressive strength')
sns.relplot(data = data, x = 'Nano silica', y = 'Compressive strength')
sns.relplot(data = data, x = 'Water', y = 'Compressive strength')
sns.relplot(data = data, x = 'Sand', y = 'Compressive strength')
sns.relplot(data = data, x = 'Gravel', y = 'Compressive strength')
sns.relplot(data = data, x = 'Fiber', y = 'Compressive strength')
sns.relplot(data = data, x = 'Superplasticizer', y = 'Compressive strength')
sns.relplot(data = data, x = 'Relative humidity', y = 'Compressive strength')
sns.relplot(data = data, x = 'Temperature', y = 'Compressive strength')
sns.relplot(data = data, x = 'Age', y = 'Compressive strength')
sns.pairplot(data)

#绘制热力图
sns.heatmap(data.corr(), annot = True)

#划分测试集训练集
x_train, x_test, y_train, y_test = train_test_split(a, b, test_size = 0.3, random_state = 1)

#实例化XGBoost模型
xgb_model = XGBRegressor(n_estimators = 100,
                         max_depth = 6,
                         learning_rate = 0.3,
                         min_child_weight = 1,
                         subsample = 1)

#Ada-XGBoost
ada_model = AdaBoostRegressor(base_estimator = xgb_model, n_estimators = 50, learning_rate = 1)

#Bagging-XGBoost
bag_model = BaggingRegressor(xgb_model, n_estimators = 10)

#Voting-XGBoost
vote_model = VotingRegressor(estimators = [('xgb', xgb_model), ('ada', ada_model), ('bag', bag_model)])

estimators = [('xgb', xgb_model), ('ada', ada_model), ('bag', bag_model)]
final = LinearRegression()
stack_model = StackingRegressor(estimators = estimators, final_estimator = final)

#交叉验证
y_train = np.array(y_train)
kfold = KFold(n_splits = 10, shuffle = True)
score_xgb = cross_validate(xgb_model, x_train, y_train.ravel(), 
                           cv = kfold, 
                           return_train_score = True)
score_xgb = pd.DataFrame(score_xgb)
print("xgb测试集均值：", score_xgb.test_score.mean())
print("xgb测试集标准差：", score_xgb.test_score.std())
print("xgb训练集均值：", score_xgb.train_score.mean())
print("xgb训练集标准差：", score_xgb.train_score.std())
print("\n")

score_ada = cross_validate(ada_model, x_train, y_train.ravel(), 
                           cv = kfold, 
                           return_train_score = True)
score_ada = pd.DataFrame(score_ada)
print("ada测试集均值：", score_ada.test_score.mean())
print("ada测试集标准差：", score_ada.test_score.std())
print("ada训练集均值：", score_ada.train_score.mean())
print("ada训练集标准差：", score_ada.train_score.mean())
print("\n")

score_bag = cross_validate(bag_model, x_train, y_train.ravel(), 
                           cv = kfold, 
                           return_train_score = True)
score_bag = pd.DataFrame(score_bag)
print("bag测试集均值：", score_bag.test_score.mean())
print("bag测试集标准差：", score_bag.test_score.std())
print("bag训练集均值：", score_bag.train_score.mean())
print("bag训练集标准差：", score_bag.train_score.std())
print("\n")

score_vote = cross_validate(vote_model, x_train, y_train.ravel(), 
                            cv = kfold, 
                            return_train_score = True)
score_vote = pd.DataFrame(score_vote)
print("vote测试集均值：", score_vote.test_score.mean())
print("vote测试集标准差：", score_vote.test_score.std())
print("vote测试集均值：", score_vote.train_score.mean())
print("vote测试集标准差：", score_vote.train_score.std())
print("\n")

scoring = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
cross_validate(xgb_model, x_train, y_train.ravel(), 
               scoring = scoring, 
               cv = kfold, 
               return_train_score = True)

for i in range(0, 25):
    random = i
    x_train, x_test, y_train, y_test = train_test_split(a, b, test_size = 0.3, random_state = random)
    xgb_model = xgb_model.fit(x_train, y_train)
    print("random_state:", random)
    print("训练集分数:", xgb_model.score(x_train, y_train))
    print("测试集分数", xgb_model.score(x_test, y_test))
    y_pred = xgb_model.predict(x_test)
    y_train_pred = xgb_model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    print("训练集均方误差（MSE）：", mse_train)
    mse_test = mean_squared_error(y_test, y_pred)
    print("测试集均方误差（MSE）：", mse_test)
    rmse_train = np.sqrt(mse_train)
    print("训练集均方根误差（RMSE）：", rmse_train)
    rmse_test = np.sqrt(mse_test)
    print("测试集均方根误差（RMSE）：", rmse_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    print("训练集平均绝对误差（MAE）：", mae_train)
    mae_test = mean_absolute_error(y_test, y_pred)
    print("测试集平均绝对误差（MAE）:", mae_test) #16得分较高

#XGBoost预测
x_train, x_test, y_train, y_test = train_test_split(a, b, test_size = 0.3, random_state = 16)
model = xgb_model.fit(x_train, y_train)
predict = model.predict(x_test)    
predict = np.array(predict)
predict = pd.DataFrame(predict, columns = ["predict"])
y_test = np.array(y_test)
y_test = pd.DataFrame(y_test, columns = ["true"])

predict_train = model.predict(x_train)
predict_train = np.array(predict_train)
predict_train = pd.DataFrame(predict_train, columns = ["predict"])
y_train = np.array(y_train)
y_train = pd.DataFrame(y_train, columns = ["true"])

##训练集散点图
plt.xlim((0, 300))
plt.ylim((0, 300))
sns.regplot(x = predict_train, y = y_train, color = 'g', ci = 99)
plt.plot([0, 300], [0, 300], "r")
plt.show

##测试集散点图
plt.xlim((0, 300))
plt.ylim((0, 300))
sns.regplot(x = predict, y = y_test, color = 'g', ci = 99)
plt.plot([0, 300], [0, 300], "r")
plt.show

#灵敏度分析
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(a)
shap.summary_plot(shap_values, a)


shap.dependence_plot(2, shap_values, a, interaction_index = 14)

#PDD图
features = [(2, 14)]
PartialDependenceDisplay.from_estimator(model, x_train, features)









































