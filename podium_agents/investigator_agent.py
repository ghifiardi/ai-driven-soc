import random
import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_classification_engine import EnhancedClassificationEngine

class InvestigatorAgent:
    def __init__(self):
        self.engine = EnhancedClassificationEngine()
        self.active_cases = [] # For dashboard
        self.investigation_templates = {
            "fraud_pattern": self.investigate_fraud,
            "operational_issue": self.investigate_operations,
            "sentiment_spike": self.investigate_sentiment
        }
    
    def investigate_anomaly(self, alert):
        """Melakukan investigasi mendalam pada anomali"""
        anomaly_type = self.classify_anomaly_type(alert)
        
        # Pilih template investigasi berdasarkan jenis anomali
        investigation_method = self.investigation_templates.get(
            anomaly_type, 
            self.general_investigation
        )
        
        findings = investigation_method(alert)
        
        # Local "Cortex" insights using the engine's reasoning
        if findings.get("requires_deep_analysis"):
            cortex_insights = self.get_local_insights(alert)
            findings.update(cortex_insights)
        
        # Rekomendasi tindakan
        recommendations = self.generate_recommendations(findings)
        
        result = {
            "alert": alert,
            "findings": findings,
            "recommendations": recommendations,
            "priority_level": self.calculate_priority(findings),
            "identified_risks": [anomaly_type] if anomaly_type != "unknown" else [],
            "requires_immediate_action": findings.get("risk_score", 0) > 6.0 # Updated threshold for 0-10 scale
        }
        self.active_cases.append(result)
        return result
    
    def investigate_fraud(self, alert):
        """Investigasi khusus untuk pola penipuan"""
        # Create a rich alert object for the engine if needed, or use existing
        # We assume alert is dict.
        
        # Use engine to detect patterns (ThreatScore)
        threat_score = self.engine.calculate_threat_score(alert)
        
        # Map findings
        suspicious_patterns = threat_score.reasoning
        
        return {
            "suspicious_patterns": suspicious_patterns,
            "risk_score": threat_score.total_score,
            "requires_deep_analysis": threat_score.total_score > 4.0
        }
    
    # --- LOCAL LOGIC METHODS ---

    def classify_anomaly_type(self, alert):
        # deterministically classify based on metric if available
        metric = alert.get("metric", "")
        if "fraud" in metric or "dispute" in metric:
            return "fraud_pattern"
        elif "sentiment" in metric:
            return "sentiment_spike"
        elif "volume" in metric or "response" in metric:
            return "operational_issue"
        return "unknown"

    def general_investigation(self, alert):
        score = self.engine.calculate_threat_score(alert).total_score
        return {"risk_score": score, "requires_deep_analysis": False}
        
    def investigate_operations(self, alert):
        score = self.engine.calculate_threat_score(alert).total_score
        return {"risk_score": score, "requires_deep_analysis": False, "type": "ops"}

    def investigate_sentiment(self, alert):
        score = self.engine.calculate_threat_score(alert).total_score
        return {"risk_score": score, "requires_deep_analysis": True, "type": "sentiment"}

    def get_local_insights(self, alert):
        # Simulate AI insight generation by analyzing score components
        score_obj = self.engine.calculate_threat_score(alert)
        insights = []
        if score_obj.attack_category_score > 0:
            insights.append(f"High risk attack category identified.")
        if score_obj.severity_score > 0:
            insights.append("Severity indicates critical processing priority.")
            
        return {"ai_insight": " | ".join(insights) if insights else "No specific AI insights generated."}

    def find_similar_historical_cases(self, alert):
        return [{"id": "case_123", "similarity": 0.8}] # Keep mock for now
        
    def generate_recommendations(self, findings):
        score = findings.get("risk_score", 0)
        recs = []
        if score > 8.0:
            recs.append("Immediate Block")
            recs.append("Escalate to SOC Manager")
        elif score > 5.0:
            recs.append("Open Jira Ticket")
            recs.append("Monitor closely")
        else:
            recs.append("Log event")
        return recs
        
    def calculate_priority(self, findings):
        score = findings.get("risk_score", 0)
        if score > 8.0: return "critical"
        if score > 6.0: return "high"
        if score > 4.0: return "medium"
        return "low"
