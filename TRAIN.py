from ultralytics import YOLO
# Создание модели YOLO
model = YOLO("yolov8n.yaml")
#  Загрузка предварительной модели для тренирвоки
model = YOLO("yolov8n.pt")
# Перенос весов модели YAML на модель PT
results = model.train(data="dat.yaml", epochs=100, imgsz=640)