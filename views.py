import json
from flask import Flask
from flask import request
from libs.predict_purchase import predict_init, predict_level
from libs.decrypt import aes_decrypt
import os
import hashlib

app = Flask(__name__)


@app.route("/predict_purchase/init", methods=["POST"])
def predict_purchase_init():
    """初始判断玩家付费可能性"""
    dict_path = os.path.join(app.root_path, "models", "purchase_model", "purchase_init_dict.pkl")
    init_path = os.path.join(app.root_path, "models", "purchase_model", "purchase_init_pred.pkl")
    data = request.get_data().decode("utf-8")
    try:
        data = aes_decrypt(data)
    except Exception:
        return {"rate": 0}, 201
    rate = predict_init(dict_path, init_path, data)
    return {"rate": rate}


@app.route("/predict_purchase/level/<int:level>", methods=["POST"])
def predict_purchase_level(level):
    """在对应关卡判断玩家付费可能性"""
    level_path = os.path.join(app.root_path, "models", "purchase_model", "purchase_{}.pkl").format(level)
    data = request.get_data().decode("utf-8")
    print(data)
    try:
        data = aes_decrypt(data)
    except Exception:
        return {"rate": 0}, 201
    data = json.loads(data)
    print(data)
    rate = predict_level(level_path, data)
    return {"rate": rate}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




