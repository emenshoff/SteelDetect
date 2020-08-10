import requests
import cv2

img_path = "../steel/test_images/0c124b96b.jpg"

img = cv2.imread(img_path)

req_data = {"protocol_version": "1.0", "image": img}


# Запрашиваем Web-сервис, передаем в запрос параметры
resp = requests.get('http://127.0.0.1:5000/predict', params=req_data)

print(resp.text)



req_data = {"protocol_version": "2.0", "image": img}


# Запрашиваем Web-сервис, передаем в запрос параметры
resp = requests.get('http://127.0.0.1:5000/predict', params=req_data)

print(resp.text)