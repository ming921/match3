import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import joblib
import numpy as np



def load_predict(level, game_fail_count, time_use_sum, move_use_sum, add5moves_after_count, add5moves_before_count,
                 booster_use_count, coins_use_sum, shop_open_count, goods_click_count):
    """根据模型进行预测"""
    dict = {"game_fail_count": game_fail_count, "time_use_sum": time_use_sum, "move_use_sum": move_use_sum,
            "add5moves_after_count": add5moves_after_count, "add5moves_before_count": add5moves_before_count,
            "booster_use_count": booster_use_count, "coins_use_sum": coins_use_sum, "shop_open_count": shop_open_count,
                 "goods_click_count": goods_click_count}
    data = pd.DataFrame([dict])
    fill_dict = {"game_fail_count": 1, "time_use_sum": 1000, "move_use_sum": 200, "add5moves_after_count": 3,
                 "add5moves_before_count": 3, "booster_use_count": 3, "coins_use_sum": 2000, "shop_open_count": 0,
                 "goods_click_count": 0}
    data.fillna(value=fill_dict, inplace=True)
    print(data)
    if level == 20:
        rf = joblib.load("purchase_predict_20.pkl")
        y = rf.predict_proba(data)[0][1]
    else:
        rf = joblib.load("purchase_predict_50.pkl")
        y = rf.predict_proba(data)[0][1]
    print(y)
    return y


def predict_level(data, proportion, level, threshold):
    # 数据清洗
    data.dropna(subset=["user_type", "user_adid"], inplace=True)  # 删除user_type,user_adid为空的行
    time_use_med = data["time_use_sum"].median()
    data["time_use_sum"].fillna(time_use_med, inplace=True)  # 用中位数填充时间空值
    move_use_med = data["move_use_sum"].median()
    data["move_use_sum"].fillna(move_use_med, inplace=True)  # 用中位数填充步数空值
    data.fillna(value=0, axis=0, inplace=True)  # 其余列用0填充
    print("正负样本数量为：", sorted(Counter(data.iloc[:, -1]).items()))

    # 切分正负样本
    data_normal = data[data["user_type"] == "normal"]
    data_purchase = data[data["user_type"] == "purchase"]

    # 数据划分
    data_train = pd.concat([data_purchase.iloc[1:int(data_purchase.shape[0]*0.75), :],
                           data_normal.iloc[1: int(data_purchase.shape[0]*0.75), :]])

    data_test = pd.concat([data_purchase.iloc[int(data_purchase.shape[0]*0.75): -1, :],
                           data_normal.iloc[int(data_normal.shape[0]-data_purchase.shape[0]*0.25*proportion): -1, :]])

    # 划分x, y
    x_train = data_train.iloc[:, 1:-1]
    y_train = data_train.iloc[:, -1]
    x_test = data_test.iloc[:, 1:-1]
    y_test = data_test.iloc[:, -1]

    # 建模-随机森林
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    accuracy = rf.score(x_test, y_test)
    y_predict = []
    for i in rf.predict_proba(x_test):
        if i[1] >= threshold:
            y_predict.append("purchase")
        else:
            y_predict.append("normal")

    print("准确率：", accuracy)
    print(classification_report(y_true=y_test, y_pred=y_predict))
    print("预测为付费用户占比：", sorted(Counter(y_predict).items())[1][1] / len(y_predict))

    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    print("="*20)
    print("要素重要性排名")
    for i in range(8):
        print("%2d) %-*s %f" % (i + 1, 30, x_train.columns[i], importance[indices[i]]))
    print("="*20)

    # 保存模型
    joblib.dump(rf, "./models/purchase_{}.pkl".format(level))


if __name__ == '__main__':

    threshold_20 = 0.2  # 判断为正样本的阈值
    proportion_20 = 1  # 正负样本的比例
    data_20 = pd.read_csv("./data/purchase/data_20csv")
    # predict_level(data_20, proportion_20, 20, threshold_20)

    threshold_50 = 0.2
    proportion_50 = 1
    data_50 = pd.read_csv("./data/purchase/data_50.csv")
    predict_level(data_50, proportion_50, 50, threshold_50)
    # load_predict(level=50, game_fail_count=1, time_use_sum=1000, move_use_sum=200, add5moves_after_count=3, add5moves_before_count=3,
    #              booster_use_count=3, coins_use_sum=2000, shop_open_count=0, goods_click_count=0)

