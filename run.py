import threading
from backend import app


if __name__ == "__main__":
#    app.run(host="192.168.1.6", debug=True)
    app.run(host="127.0.0.1", debug=True)
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5001}).start()
