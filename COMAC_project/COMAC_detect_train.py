

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r'D:\faults_detection\ultralytics\config_develop\yolo26l-p2-ccfm.yaml')

    model.train(
        data=r'D:\faults_detection\ultralytics\COMAC_project\data.yaml', imgsz=1024, epochs=100, batch=1, workers=1, device=[0],
        optimizer='AdamW', cos_lr=True, lr0=0.0001, lrf=0.01,             
        save_period=10,cache=False,   

        project=r'D:\faults_detection\ultralytics\output\COMAC', name='yolo26_l_260428', pretrained=False,

        hsv_h=0, hsv_s=0.5, hsv_v=0.5,
        degrees=0, scale=0.5, flipud=0.5, fliplr=0.5, erasing=0.0,          
        mosaic=1.0, mixup=0, multi_scale=0,        
        
        resume=False, amp=True, exist_ok=True
    )