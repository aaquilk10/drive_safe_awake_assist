import uuid
import os
import time
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

IMAGES_PATH = os.path.join('data', 'images')
labels = ['awake', 'drowsy']
number_imgs = 5

cap = cv2.VideoCapture(0)
for label in labels:

    print('Label - {}'.format(label))
    time.sleep(5)
    
    for img_num in range(number_imgs):
        print('Label - {}, image number - {}'.format(label, img_num))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('Image', frame)
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

print(os.path.join(IMAGES_PATH, labels[0]+'.'+str(uuid.uuid1())+'.jpg'))

for label in labels:
    print('Label - {}'.format(label))
    for img_num in range(number_imgs):
        print('Label - {}, image number - {}'.format(label, img_num))
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        print(imgname)  

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp15/weights/last.pt', force_reload=True)

img = os.path.join('data', 'images', 'IMAGE_NAME')

results = model(img)

results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()