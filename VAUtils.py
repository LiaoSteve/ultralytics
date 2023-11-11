from select import select
import numpy as np
import cv2
import math
import random
import time

def getColor(idx):
    # return bgr color
    if idx == 0: # person (purple-red)
        # return (180,50,200)
        return (100,255,100)
    elif idx == 1: # bicycle (blue)
        return (255,0,0)
    elif idx == 2: # car (green)
        return (100,255,100)
    elif idx == 3: #  motorcycle (yellow)
        return (0,255,255)
    elif idx == 5: # bus (light blue)
        return (255,255,0)
    elif idx == 7: # truck (white)
        return (255,255,255)
    else:
        return (0,0,255)
#====================================================
#   Point
#====================================================
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

#====================================================
#   vector
#====================================================
def vector(pt1, pt2): 
    return (pt2.x - pt1.x, pt2.y - pt1.y)

#====================================================
#   BBox
#====================================================
class BBox():
    def __init__(self, class_id, x1, y1, x2, y2, score, contour=None, contour_area=0, max_contour=None, inactive=0, abandoned_prob=0.0):
        self.class_id = class_id
        self.tl = Point(x1,y1)
        self.br = Point(x2,y2)
        self.score = score
        self.contour = contour # for blob
        self.max_contour = max_contour
        self.contour_area = contour_area # for blob
        self.lifetime = 0
        self.inactive = inactive
        self.abandoned_prob = abandoned_prob

    def getCenterPoint(self):
        cx = (self.tl.x + self.br.x)//2
        cy = (self.tl.y + self.br.y)//2
        return Point(cx, cy)
    
    def getBottomCenterPoint(self):
        x = (self.tl.x + self.br.x)//2
        y = self.br.y
        return Point(x, y)

    def getRadius(self):
        c = self.getCenterPoint()
        r = math.sqrt(abs(c.x - self.tl.x)**2 + (abs(c.y - self.tl.y))**2)
        return r



from PIL import ImageGrab
import cv2
import numpy as np
import os
import random

class VideoSource():
    def __init__(self, source_from_cv2=True, screen_bbox=(0,0,1920,1080), video_path=None, is_image=False, start_idx=0):
        """
        source_from_cv2
        - True: webcam or video file
        - False: from live screen 
        """
        self.source_from_cv2 = source_from_cv2
        self.is_image = is_image
        self.screen_bbox = screen_bbox
        self.cap = None
        self.image_files = []
        self.image_idx = 0
        self.start_idx = start_idx
        self.sw1 = False
        self.sw2 = False
        self.use_blur = False
        self.flip = False
        self.use_bgr = True # nanodet use bgr to train

        if self.start_idx > 0:
            self.image_idx = self.start_idx - 1
                    
        if self.source_from_cv2:
            if not self.is_image:
                self.cap = cv2.VideoCapture(video_path)
                self.wait_key_ms = 1
            else:
                self.wait_key_ms = 0
                files = os.listdir(video_path)
                for file in files:
                    if file.endswith("jpg") or file.endswith("png"):
                        self.image_files.append(f"{video_path}/{file}")
        else: 
            self.cap = ImageGrab.grab(bbox=screen_bbox)
            self.is_rgb = True
            self.wait_key_ms = 1

    def random_brightness(self, img, delta_low=0.2, delta_up=0.2):
        img = img.astype(float)
        img += random.uniform(-delta_low*255, delta_up*255)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def random_contrast(self, img, alpha_low=0.6, alpha_up=1.4):
        img = img.astype(float)
        img *= random.uniform(alpha_low, alpha_up)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def random_saturation(self, img, alpha_low=0.5, alpha_up=1.2):
        hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
        hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def getFrame(self):
        """ return bgr np array uint8 image"""
        if self.source_from_cv2:
            if not self.is_image:
                ret, frame = self.cap.read()
                return ret, frame, None
            else:
                try:
                    path = self.image_files[self.image_idx]
                    print(f"idx: {self.image_idx}/{len(self.image_files)} | {self.image_idx/len(self.image_files)*100:.2f}%")
                    image = cv2.imread(path)
                    # image = cv2.resize(image, None, fx=1.5, fy=1.5)
                    self.image_idx += 1
                    return True, image, path
                except:
                    print("No images")
                    return False, None, None
        else:
            frame = np.array(ImageGrab.grab(bbox=self.screen_bbox)) # rgb
            # frame = cv2.resize(frame, (1440, 810))
            if self.use_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # bgr

            if self.sw2:
                frame = self.random_brightness(frame)
                frame = self.random_contrast(frame)
                frame = self.random_saturation(frame)

                # h, w, c = frame.shape
                # frame = frame[int(h*0):int(h*1), int(w*0.25):int(w*0.75)]
            
            if self.flip:
                frame = cv2.flip(frame, 0)
                # h, w, c = frame.shape
                # frame = frame[int(h*0):int(h*1), int(w*0.25):int(w*0.75)]
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
            # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 7, 21) # 2 DNR

            if self.use_blur:
                frame = cv2.blur(frame, (5, 5))

            # h, w, c = frame.shape
            # noise = np.random.normal(0, 0.1, (h, w, c)).astype(np.uint8)
            # frame = cv2.add(frame, noise)

            return True, frame, None
    
    def close(self):
        if self.source_from_cv2:
            if not self.is_image:
                self.cap.release()

class FPSController:
    def __init__(self, fps=30):
        self._fps = fps
        self.interval = 1.0 / fps
        self.last_time = time.monotonic()
        self.elapsed_time = 0.0
        self._cur_fps = 0.01

    def setFPS(self, fps):
        self._fps = fps
        self.interval = 1.0 / fps
        
    def tick(self, show_info=False):
        now = time.monotonic()
        self.elapsed_time = now - self.last_time
        self._cur_fps = 1 / (self.elapsed_time + 0.001)
        delay = self.interval - self.elapsed_time
        if show_info:
            print(f"SYS_FPS:{self._fps} | CUR_FPS:{self._cur_fps:.2f} | TIME:{self.elapsed_time:.2f} | DELAY:{delay:.2f}(s)")
        if delay > 0:
            time.sleep(delay)
            self.last_time += self.interval
        else:
            self.last_time = now

    def getCurFPS(self):
        return self._cur_fps
    
    def getElapsedTime(self):
        return self.elapsed_time