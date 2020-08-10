import flask
import numpy as np
from detection import Detection
import time


protocol_version = "1.0"

# Загрузка модели
detector = Detection(load_weights_from="./steel/model_unet.dat")


app = flask.Flask(__name__)


# predict function
@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    формат обмена:
    в запросе GET должен указываться параметр image со значением данных картинки (в формате массива)
    в ответе прилетает success , диагноз в текстовом параметре prediction и сегментированная картинка в
    параметре prediction_image

    """
    response_data = {"success": False, "prediction": None, "prediction_image": None}
    params = flask.request.args
    if (params == None):
        response_data["comment"] = "wrong request format"

    # Если параметры найдены, и ест данные картинки
    elif params["protocol_version"] != protocol_version:
        response_data["comment"] = "wrong protocol version"
    elif params["image"] != None:
        try:
            img = np.asarray(params["image"])
            t1 = time.time()
            img_result, description = detector.process_image(img)
            t2 = time.time()
            response_data["prediction"] = description
            response_data["prediction_image"] = img_result
            response_data["success"] = True
            response_data["comment"] = f"processed for {t2 - t1} msec"
        except Exception:
            response_data["comment"] = f"wrong request format. Exception: {str(Exception)}"

    else:
        response_data["comment"] = "wrong request format"

    # Возвращаем результат в json format
    return flask.jsonify(response_data)





