from detection import Detection


def callback():
    print("Custom callback called!")

model = Detection()

model.fit_model(fit_callback=callback)