from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# model = YOLO('yolov8n.yaml')
# print(model)
# result = model.predict("https://ultralytics.com/images/bus.jpg", save=True, save_txt=True, iou=0.3, conf=0.55, max_det=100, imgsz=640)
# result = model.predict("bus.jpg", save=True, save_txt=True, iou=0.3, conf=0.55, max_det=100, imgsz=640)
path = model.export(format="onnx", simplify=True, imgsz=[640,640])  # export the model to ONNX format