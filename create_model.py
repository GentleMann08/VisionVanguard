from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO('yolov8n.pt')

model.train(data="GuardianAI\data.yaml", epochs=75)