import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO
import shutil
import glob
from PIL import Image
import os



os.mkdir(r'C:\Users\nijat\OneDrive\Desktop\projects\ufaz-capstone\data\nose_x')
os.mkdir(r'C:\Users\nijat\OneDrive\Desktop\projects\ufaz-capstone\data\nose_y')

xs = glob.glob(r'C:\Users\nijat\OneDrive\Desktop\projects\ufaz-capstone\data\x\*.png')
ys = glob.glob(r'C:\Users\nijat\OneDrive\Desktop\projects\ufaz-capstone\data\y\*.png')

print(len(xs),len(ys))

for path in xs:
    
    img_title = path.split('\\')[-1]
    
    
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (640, 640))
    
    try:
    
        model = YOLO('model.pt')
    
    
        results = model.predict(img)
        
        
        coor = results[0].boxes.xyxy
    
        arr = img
        coors = coor[0]
        x1 = int(float(coors[0]))
        y1 = int((float(coors[1])) )
        x2 = int((float(coors[2])) )
        y2 = int((float(coors[3])) )
        
        
        
        new_arr = arr[y1:y2,x1:x2]
    except IndexError:
        continue
    
    
    new_img = Image.fromarray(new_arr)
    new_img.save(r'C:\Users\nijat\OneDrive\Desktop\projects\ufaz-capstone\data\nose_x\\'+img_title)
    
for path in ys:
    
    img_title = path.split('\\')[-1]
    
    
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (640, 640))
    
   
    
    try:
    
        model = YOLO('model.pt')
    
    
        results = model.predict(img)
        
        
        coor = results[0].boxes.xyxy
    
        arr = img
        coors = coor[0]
        x1 = int(float(coors[0]))
        y1 = int((float(coors[1])) )
        x2 = int((float(coors[2])) )
        y2 = int((float(coors[3])) )
        
        
        
        new_arr = arr[y1:y2,x1:x2]
    except IndexError:
        continue
    
    
    new_img = Image.fromarray(new_arr)
    new_img.save(r'C:\Users\nijat\OneDrive\Desktop\projects\ufaz-capstone\data\nose_y\\'+img_title)
        
    