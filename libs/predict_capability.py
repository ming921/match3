import gc
import pandas as pd
import joblib


def predict_capability_level(std_path, level_path, dict):
    """到指定关卡时预测"""
    data = pd.DataFrame([dict])
    fill_dict = {"time_sum": 11462.1, "moves": 1655.7, "fail_num": 13.6, "front_booster_num": 45.5, "booster_num": 16.0,
                 "add5stepsleft_num": 15.4, "add5steps_show_num": 12.8, "add5steps_num": 5.1, "gain_booster_sum": 398.9,
                 "per_dessert_count": 258.3, "per_effect_num": 22.2}
    data.fillna(value=fill_dict, inplace=True)
    std = joblib.load(std_path)
    data = std.transform(data)
    model = joblib.load(level_path)
    kind = model.predict(data)[0]

    return kind


