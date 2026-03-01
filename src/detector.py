import cv2
import os
import torch
from ultralytics import YOLO

class FireDetector:
    def __init__(self, model_path='yolov8s-cls.pt'):
        """
        Initialize the Advanced Fire Vision Engine.
        Prioritizing YOLOv8 Small (S) for higher precision over Nano.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Advanced Vision Engine (YOLOv8s) on {self.device}...")
        
        self.reload_model()

    def reload_model(self):
        """
        Reloads the best weights, prioritizing the user-retrained High Precision model.
        """
        # 1. Check for Retrained Model (User Feedback Loop)
        retrained_path = 'runs/classify/ignisguard_retrained_model/weights/best.pt'
        # 2. Check for Advanced Model (Pre-shipped)
        advanced_path = 'runs/classify/ignisguard_advanced_model/weights/best.pt'
        
        if os.path.exists(retrained_path):
            print(f"🔥 Loading CUSTOM RETRAINED 'FIRE BRAIN' Weights: {retrained_path}")
            self.model = YOLO(retrained_path)
        elif os.path.exists(advanced_path):
            print(f"Loading Advanced Weights: {advanced_path}")
            self.model = YOLO(advanced_path)
        else:
            print("Using Generic YOLOv8s-cls Weights.")
            self.model = YOLO('yolov8s-cls.pt')
        
    def detect(self, source, conf=0.30):
        """
        Detect/Classify fire/smoke. 
        Relaxed confidence filter (0.30) to maximize RECALL.
        We let the Decision Engine handle the final filtering.
        """
        results = self.model(source, device=self.device)
        
        # Classification output
        if hasattr(results[0], 'probs'):
            probs = results[0].probs
            top1_idx = probs.top1
            label = results[0].names[top1_idx]
            confidence = float(probs.top1conf)
            
            # RELAXED FILTER: Lowered floor to catch faint smoke
            if confidence < conf:
                return [] 
                
            return [{"class": label, "confidence": confidence}]
        
        return []

    def process_video(self, video_path, output_path='data/uploads/drone/output.mp4', decision_engine=None):
        """
        Process drone video with Temporal Consistency check.
        Features:
        - Frame Skipping (5 frames) for performance.
        - Custom Alert Overlays (Red Border/Text for Fire).
        """
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Performance: Process every Nth frame
        FRAME_SKIP = 5
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_data = [] # Stores results for temporal analysis (sampled frames)
        frame_count = 0
        last_results = None # Persist last detection for skipped frames
        
        print(f"Applying Temporal Analysis to: {video_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- INTELLIGENT SAMPLING ---
            is_processed_frame = (frame_count % FRAME_SKIP == 0)
            
            if is_processed_frame:
                results = self.model(frame, verbose=False)
                last_results = results
                
                # Extract classification for temporal log
                probs = results[0].probs
                if probs:
                    label = results[0].names[probs.top1]
                    conf = float(probs.top1conf)
                    frame_data.append([{"class": label, "confidence": conf}])
            
            # --- CUSTOM VISUALIZATION (Overlay) ---
            # Re-use last_results for skipped frames to maintain visual continuity
            if last_results and last_results[0].probs:
                probs = last_results[0].probs
                label = last_results[0].names[probs.top1]
                conf = float(probs.top1conf)
                
                is_fire = (label == 'fire' and conf > 0.5) # Threshold check
                
                # overlay setup
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 255) if is_fire else (0, 255, 0) # Red or Green
                text = f"ALERT: FIRE DETECTED [{conf:.2f}]" if is_fire else f"STATUS: CLEAR [{conf:.2f}]"
                
                if is_fire:
                    # Draw Thick Red Border
                    cv2.rectangle(frame, (0, 0), (width, height), color, 20)
                    # Draw Text Background
                    cv2.rectangle(frame, (50, 50), (600, 150), (0, 0, 0), -1)
                    cv2.putText(frame, "CRITICAL WARNING", (70, 90), font, 1, (0, 0, 255), 2)
                    cv2.putText(frame, text, (70, 130), font, 0.8, (255, 255, 255), 2)
                else:
                    # Draw Subtle Green Status
                    cv2.rectangle(frame, (20, 20), (450, 80), (0, 0, 0), -1)
                    cv2.putText(frame, text, (40, 60), font, 0.8, (0, 255, 0), 2)

            out.write(frame)
            frame_count += 1
            
        cap.release()
        out.release()
        
        # Final Decision via Temporal Verification
        temporal_verified = False
        if decision_engine:
            # Note: We are verifying based on the SAMPLED sequence
            temporal_verified, ratio = decision_engine.verify_temporal_consistency(frame_data)
            print(f"Temporal Consistency Score: {ratio:.2f} (Sampled)")

        return output_path, temporal_verified

    def process_satellite_imagery(self, bands):
        """
        Placeholder for multi-spectral analysis (SWIR/NIR).
        In a real scenario, this would involve processing .tiff files and
        detecting thermal anomalies.
        """
        # Logic for thermal anomaly detection (e.g., Normalized Burn Ratio)
        pass

if __name__ == "__main__":
    # Quick test initialization
    detector = FireDetector()
    print("Fire Detector Initialized Successfully.")