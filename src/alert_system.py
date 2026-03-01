import json
import os
from datetime import datetime

class AlertManager:
    def __init__(self, log_file="data/alert_log.json"):
        self.log_file = log_file
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Init log file if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def evaluate_risk(self, risk_score, wind_speed, location):
        """
        Evaluates risk parameters against thresholds to determine Alert Level.
        Returns: Dict with status ('SAFE', 'WARNING', 'CRITICAL') and message.
        """
        status = "SAFE"
        message = "Risk levels are within normal limits."
        
        # Rule Set
        if risk_score > 0.75:
            status = "CRITICAL"
            message = "EXTREME FIRE DANGER. Immediate preventive action recommended."
        elif risk_score > 0.6 or (risk_score > 0.4 and wind_speed > 30):
            status = "WARNING"
            message = "High fire danger conditions detected. Monitor closely."
        
        if status != "SAFE":
            self.log_alert({
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "location": location,
                "risk_score": risk_score,
                "wind_speed": wind_speed,
                "message": message
            })
            
        return {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

    def log_alert(self, alert_data):
        """
        Logs the alert to a JSON file (simulating DB/Notification Dispatch).
        """
        try:
            with open(self.log_file, 'r+') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
                
                logs.append(alert_data)
                f.seek(0)
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"Alert Logging Error: {e}")

    def get_recent_alerts(self, limit=10):
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
                return logs[-limit:][::-1]
        except:
            return []
