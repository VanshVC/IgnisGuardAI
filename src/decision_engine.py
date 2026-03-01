import os

class DecisionEngine:
    def __init__(self, confidence_threshold=0.85):
        """
        Initialize the Intelligent Decision Engine.
        Goal: Reduce false positives by fusing multi-source data.
        """
        self.confidence_threshold = confidence_threshold

    def verify_detection(self, vision_results, satellite_risk=None):
        """
        Fuses AI Vision (YOLOv8) with Geospatial Context (GEE).
        - Temporal consistency: Handled in video processor.
        - Cross-check: Correlates ground vision with satellite NBR/NDVI.
        """
        # Extract Vision Data
        fire_detections = [d for d in vision_results if d['class'] == 'fire']
        primary_confidence = max([d['confidence'] for d in fire_detections], default=0)
        
        # 1. Vision Check (BALANCED BASE)
        # Relaxed from 0.90 to 0.70 to ensure we catch early-stage fires
        is_fire_vision = primary_confidence >= 0.70
        
        # 2. Satellite Risk Cross-Check (Correlation)
        sat_boost = False
        risk_level = satellite_risk.get('risk_level', 'LOW') if satellite_risk else 'LOW'
        
        if risk_level == 'HIGH':
            sat_boost = True
            # Critical Risk Area: Significant drop in required confidence (0.55)
            # Logic: If satellite says "High Risk", even faint smoke (55%) is likely fire.
            if primary_confidence >= 0.55:
                is_fire_vision = True
        elif risk_level == 'MEDIUM':
            sat_boost = True
            # Moderate Risk: 0.60
            if primary_confidence >= 0.60:
                is_fire_vision = True
        
        # 3. Environmental Context Check (New)
        env_factors = satellite_risk.get('environmental_factors', {}) if satellite_risk else {}
        temp = env_factors.get('temperature', 25)
        humidity = env_factors.get('humidity', 50)
        
        if temp > 35 and humidity < 35:
            # Hot & Dry conditions typically mean real fire
            if primary_confidence >= 0.50:
                is_fire_vision = True
                sat_boost = True 

        # 3. Final Decision Logic
        alert_status = "NORMAL"
        if is_fire_vision:
            alert_status = "CRITICAL_VERIFIED"
        elif primary_confidence > 0.4:
            alert_status = "WARNING_SUSPICIOUS"
            
        return {
            "is_verified": is_fire_vision,
            "alert_status": alert_status,
            "confidence_score": round(primary_confidence, 4),
            "satellite_correlated": sat_boost,
            "risk_context": risk_level
        }

    def verify_temporal_consistency(self, frame_results, window_size=10, min_ratio=0.7):
        """
        Applies Temporal Consistency check for drone/video feeds.
        Fire must be detected in at least 70% of the frames in the sliding window
        to trigger a high-confidence alert.
        """
        if len(frame_results) < window_size:
            window = frame_results
        else:
            window = frame_results[-window_size:]
            
        fire_frames = sum([1 for res in window if any(d['class'] == 'fire' for d in res)])
        ratio = fire_frames / len(window)
        
        return ratio >= min_ratio, ratio