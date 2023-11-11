import torch
import time
import cv2
from ultralytics import YOLO
from VAUtils import*
    
class ObjectDetectionYOLO(object):
    def __init__(self, 
                 model_idx = 0,
                 allowed_classes = range(79) , # coco 80 classes
                 device = "cuda"if torch.cuda.is_available() else "cpu",
                 score_thresh=0.5,
                 iou=0.45,
                 imgsz=640,
                 half = False):
        
        model_names = ["yolov8n.pt", 
                       "yolov8s.pt",
                       "yolov8m.pt",
                       "yolov8l.pt",
                       "yolov8x.pt"]
        # model_names = ["yolov8n", 
        #                "yolov8s",
        #                "yolov8m",
        #                "yolov8l",
        #                "yolov8x"]
        
        self.device = device
        self.iou = iou
        self.imgsz = imgsz
        self.half = half
        self.score_thresh = score_thresh
        self.allowed_classes = allowed_classes

        print(f"Using Device: {self.device}")
        self.model = self.__loadModel(model_names[model_idx])

    def __loadModel(self, model_path):
        print(f"Load Model {model_path}")
        model = YOLO(model_path)  # load a pretrained model (recommended for training)
        # model = YOLO(f'{model_path}.yaml').load(f'{model_path}.pt')
        model.fuse()

        return model

    def detect(self, frame):
        # return batch result
        # https://docs.ultralytics.com/modes/predict/#inference-sources
        return True, self.model(frame, 
                                save = False, 
                                save_txt = False, 
                                iou = self.iou, 
                                conf = self.score_thresh, 
                                max_det = 100, 
                                imgsz = self.imgsz,
                                classes = self.allowed_classes, 
                                half = self.half,
                                device = self.device,
                                )
    
    def processResultBatchProces(self, batch_results): # To do: support batch process
        pass
        # all_boxes = []
        # for batch, results in enumerate(batch_results):            
        #     xyxys = results.boxes.xyxy.cpu().numpy()
        #     classes = results.boxes.cls.cpu().numpy()
        #     confs = results.boxes.conf.cpu().numpy()
            
        #     for i, xyxy in enumerate(xyxys):
        #         conf = confs[i]
        #         class_idx = classes[i]

        #         if conf < self.score_thresh:
        #             continue

        #         all_boxes.append(
        #             BBox(int(class_idx), 
        #                     int(xyxy[0]), 
        #                     int(xyxy[1]),
        #                     int(xyxy[2]),
        #                     int(xyxy[3]),
        #                     conf
        #                     ))
        # return all_boxes 
    
    def processResult(self, results): # only one batch
        all_boxes = []
        xyxys = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        for i, xyxy in enumerate(xyxys):
            conf = confs[i]
            class_idx = classes[i]

            if conf < self.score_thresh:
                continue

            all_boxes.append(
                BBox(int(class_idx), 
                        int(xyxy[0]), 
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                        conf
                        ))
        return all_boxes 

if __name__ == '__main__':
    class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
                'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
                'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
    
    # od = ObjectDetectionYOLO(model_idx=1, allowed_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], score_thresh=0.45)
    od = ObjectDetectionYOLO(model_idx=0, allowed_classes=[0, 1, 2, 3, 5, 7], score_thresh=0.4, imgsz=640)
    # od = ObjectDetectionYOLO(model_idx=1, allowed_classes=[0,1,2,3,5,7,14,15,16,17,18,19,20,21,22,23], score_thresh=0.25, imgsz=640)
    # video_source = VideoSource(source_from_cv2=False)
    video_source = VideoSource(video_path=".", is_image=True)
    G_fps_ctrl = FPSController(fps=30)

    while True:
        ret, frame, image_path = video_source.getFrame()
        if not ret: break

        save_frame = frame.copy()
        _, res = od.detect(frame)
        bboxes = od.processResult(res[0]) # new method
        yolo_bboxes = []

        for i, bbox in enumerate(bboxes):
            text = f"{class_names[bbox.class_id]} {int(bbox.score * 100)}%"
            color = getColor(bbox.class_id)
            cv2.rectangle(frame, (bbox.tl.x, bbox.tl.y), (bbox.br.x, bbox.br.y), color, 2)
            cv2.putText(frame, text, (bbox.tl.x+3, bbox.tl.y+13),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (bbox.tl.x+3, bbox.tl.y+13),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"fps: {G_fps_ctrl.getCurFPS():.2f}", (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("NanoDet in real-time ", frame)
        key = cv2.waitKey(video_source.wait_key_ms)
        G_fps_ctrl.tick(show_info=0)

        if key == 27: 
            break