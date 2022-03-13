import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc

import warnings
warnings.filterwarnings("ignore")


RATE = 617  # 617
THRESHOLD = 0.5


def load_predict(dict):
    """根据模型进行预测"""
    data = pd.DataFrame([dict])
    ios_modes = ["iPhone", "iPad"]
    data["platform"] = "ios" if any(i in data["device_info_mobile_model"][0] for i in ios_modes) else "android"
    print(data)
    fill_dict = {"platform": "android", "device_info_mobile_model": "SM-A515F", "device_info_ram": 1581,
                 "user_property_af_status": "non-organic", "user_property_af_channel": "unity",
                 "geo_country": "PK", "first_cpm": 20}
    data.fillna(value=fill_dict, inplace=True)
    dict = joblib.load("./models/purchase_init_dict.pkl")
    data = dict.transform(data.to_dict(orient="records"))

    rf = joblib.load("./models/purchase_init_pred.pkl")
    rate = rf.predict_proba(data)[0][1]
    print(rate)
    return rate


def predict():

    # 读取android&ios数据
    data = pd.read_csv("./data/purchase/data_init.csv")

    # 数据清洗
    data.dropna(subset=["user_pseudo_id", "user_type"], inplace=True)
    data.fillna(method="pad", axis=0, inplace=True)

    # 数据预处理
    # 1.划分训练集与测试集
    x = data.iloc[:, 1:-2]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # 2.样本均衡
    rus = RandomUnderSampler(random_state=0)
    x_train, y_train = rus.fit_resample(x_train, y_train)
    x_test, y_test = rus.fit_resample(x_test, y_test)
    print(x_train.columns)
    print(sorted(Counter(y_train).items()))
    print(sorted(Counter(x_train["platform"]).items()))

    # 3.文本特征抽取
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 模型选择
    rf = RandomForestClassifier()

    # 模型优化   网格搜索与交叉验证
    params = {"n_estimators": [80, 120, 200], "max_depth": [30, 50, 100]}
    gs = GridSearchCV(estimator=rf, param_grid=params, cv=4)
    gs.fit(x_train, y_train)


    # 模型评估
    y_predict_proba = gs.predict_proba(x_test)
    y_predict = []
    for i in y_predict_proba:
        if i[1] >= THRESHOLD:
            y_predict.append("purchase")
        else:
            y_predict.append("normal")
    print(classification_report(y_test, y_predict))
    print("预测为付费用户占比：", sorted(Counter(y_predict).items())[1][1]/len(y_predict))

    # 模型分析
    # 1.特征重要性排序
    importance = gs.best_estimator_.feature_importances_
    indices = np.argsort(importance)[::-1]
    for i in range(20):
        print("%2d) %-*s %f" % (i + 1, 30, dict.get_feature_names_out()[indices[i]], importance[indices[i]]))

    # 2.绘制ROC曲线
    plt.rcParams['font.sans-serif'] = ['Songti SC']
    y_score = gs.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score, pos_label="purchase")  ###计算真正率和假正率
    plt.plot(fpr, tpr, lw=2, label='ROC')
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('付费预测ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")

    # # 保存模型
    joblib.dump(dict, "./models/purchase_init_dict.pkl")
    joblib.dump(gs, "./models/purchase_init_pred.pkl")


if __name__ == '__main__':
    # predict()

    # android:SM-A515F    ios:iPhone12,1
    data = {"device_info_mobile_model": "iPhone12,1", "device_info_ram": 1581, "user_property_af_status": "non-organic",
                 "user_property_af_channel": "unity", "geo_country": "US", "first_cpm": 20}
    load_predict(data)


