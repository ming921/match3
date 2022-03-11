import pandas as pd
import joblib
import os


def predict_init(dict_path, init_path, dict):
    """初始预测"""
    data = pd.DataFrame([dict])
    fill_dict = {"platform": "android", "device_info_ram": 1581, "user_property_af_status": "non-organic",
                 "user_property_af_channel": "unity", "geo_country": "PK", "first_cpm": 20}
    data.fillna(value=fill_dict, inplace=True)
    dict = joblib.load(dict_path)
    data = dict.transform(data.to_dict(orient="records"))

    rf = joblib.load(init_path)
    rate = rf.predict_proba(data)[0][1]
    return rate


def predict_level(level_path, dict):
    """到指定关卡时预测"""
    data = pd.DataFrame([dict])
    fill_dict = {"game_fail_count": 1, "time_use_sum": 1000, "move_use_sum": 200, "add5moves_after_count": 3,
                 "add5moves_before_count": 3, "booster_use_count": 3, "coins_use_sum": 2000, "shop_open_count": 0,
                 "goods_click_count": 0}
    data.fillna(value=fill_dict, inplace=True)
    model = joblib.load(level_path)
    rate = model.predict_proba(data)[0][1]
    return rate

