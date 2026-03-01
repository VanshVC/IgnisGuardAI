import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import os

"""
IgnisGuard AI: Model Selection Analysis
---------------------------------------
As an IIT M.Tech Engineer, we evaluate two primary architectures:

1. CNN (e.g., ResNet50):
   - Pros: Excellent for classification (Fire vs No Fire).
   - Cons: No spatial localization (can't tell WHERE the fire is). Slow for high-res drone footage.

2. YOLO (e.g., YOLOv8/v10):
   - Pros: Real-time (70+ FPS). Concurrent detection and localization.
   - Cons: Requires bounding box labels (more complex data prep).

DECISION: We prioritize YOLOv8 for its real-time 'Command Center' capabilities and edge-compatibility.
"""

class FireTrainer:
    def __init__(self, data_dir='data/processed', model_type='yolov8m-cls.pt'):
        """
        Initialize the Advanced YOLOv8-Medium Classification Trainer.
        'M' version offers deeper layers for better feature extraction (smoke vs cloud).
        """
        self.model = YOLO(model_type)
        self.data_dir = data_dir

    def train(self, epochs=150, imgsz=224, batch=16):
        """
        Start the Advanced YOLOv8 training with Data Augmentation.
        Enhancements:
        - hsv_h/s/v: Restored color sensitivity (Fire is vibrant!).
        - Mosaic: Max context to understand 'Fire in Forest' vs 'Fire in Field'.
        """
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"Starting RETRAINING Sequence on {device} (High Recall Mode)...")
        
        results = self.model.train(
            data=self.data_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='ignisguard_retrained_model_v2',
            device=device,
            # Enhanced Augmentation for RECALL
            hsv_h=0.015,  # Slight hue shift allowed
            hsv_s=0.6,    # Medium saturation (catch bright flames)
            hsv_v=0.4,    # Value
            degrees=10.0, 
            flipud=0.5,   
            fliplr=0.5,   
            mosaic=1.0,   # Full Mosaic enabled for context
            mixup=0.1,    
            # Optimization
            patience=20,  # Give it time to converge
            lr0=0.001     
        )
        return results

    def export_model(self, format='onnx'):
        """
        Export for mobile/edge deployment.
        """
        self.model.export(format=format)

if __name__ == "__main__":
    # Path to the directory containing 'train' and 'val' subfolders
    DATA_PATH = os.path.abspath('data/processed')
    trainer = FireTrainer(data_dir=DATA_PATH)
    trainer.train(epochs=200) # Extended training for maximum precision