import json
import logging
from flask import Flask
from flask import request
from libs.predict_purchase import predict_init, predict_level
from libs.predict_capability import predict_capability_level
from logging import FileHandler
from libs.decrypt import aes_decrypt
import os
import gc

app = Flask(__name__)

# 定义模型存放位置
DICT_PATH = os.path.join(app.root_path, "models", "purchase_model", "purchase_init_dict.pkl")
INIT_PATH = os.path.join(app.root_path, "models", "purchase_model", "purchase_init_pred.pkl")


@app.route("/predict_purchase/init", methods=["POST"])
def predict_purchase_init():
    """初始判断玩家付费可能性"""
    data = request.get_data().decode("utf-8")
    try:
        data = aes_decrypt(data)
        data = json.loads(data)
    except Exception:
        return {"rate": 0}, 201
    rate = predict_init(DICT_PATH, INIT_PATH, data)
    app.logger.info(data)
    app.logger.info(rate)
    del data
    gc.collect()
    return {"rate": rate}


@app.route("/predict_purchase/level/<int:level>", methods=["POST"])
def predict_purchase_level(level):
    """在对应关卡判断玩家付费可能性"""
    # data = request.get_data().decode("utf-8")
    # level_path = os.path.join(app.root_path, "models", "purchase_model", "purchase_{}.pkl").format(level)
    # try:
    #     # data = aes_decrypt(data)
    #     pass
    # except Exception:
    #     return {"rate": 0}, 201
    # data = json.loads(data)
    # print(data)
    # rate = predict_level(level_path, data)
    rate = 0.7
    return {"rate": rate}


@app.route("/predict_capability/<string:system>/level/<int:level>", methods=["POST"])
def predict_capability(system, level):
    """预测玩家通关能力分层"""

    if system != "android":
        return {"kind": -1}, 202
    data = request.get_data().decode("utf-8")
    std_path = os.path.join(app.root_path, "models", "capability_model", "{}_std.pkl").format(system)
    level_path = os.path.join(app.root_path, "models", "capability_model", "{}_level_{}.pkl").format(system, level)
    try:
        # data = aes_decrypt(data)
        pass
    except Exception:
        return {"kind": -1}, 201
    data = json.loads(data)
    kind = float(predict_capability_level(std_path, level_path, data))
    print(kind)

    # 记录日志
    app.logger.info(data)
    app.logger.info(kind)

    # 删除无用缓存
    del data
    gc.collect()
    return {"kind": kind}


if __name__ == '__main__':
    app.debug = True
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    handler = FileHandler("./logs/match3.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    logging_format = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)s - %(message)s")
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port=5000, threaded=True)  # 开启多线程




