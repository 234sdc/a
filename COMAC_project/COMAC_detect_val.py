from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\faults_detection\ultralytics\COMAC_project\models\yolo26l_p2_ccfm_260429.pt")

    metrics = model.val(
        data=r"D:\faults_detection\ultralytics\COMAC_project\data.yaml",
        imgsz=1024,
        batch=8,
        workers=8,
        device=[0],
        iou=0.5,
        conf=0.01,
        end2end=False,
        project=r"D:\faults_detection\ultralytics\COMAC_project\val_outputs",
        name="yolo26l_p2_ccfm_260506",
    )
