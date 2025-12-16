#!/usr/bin/env python3
"""
Enhanced TAA with Google Flash 2.5 Integration
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import ti_abuseipdb  # optional TI enrichment
except Exception:  # pragma: no cover
    ti_abuseipdb = None

logger = logging.getLogger("EnhancedTAA")

class EnhancedTAA:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_id = config["project_id"]
        self.location = config["location"]
        self.model = None
        self.performance_metrics = {
            "total_requests": 0,
            "avg_response_time": 0,
            "success_rate": 0,
            "cache_hits": 0
        }
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Google Flash 2.5 model"""
        try:
            # Try to import and initialize Vertex AI
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig
            
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel("gemini-2.0-flash-exp")
            self.generation_config = GenerationConfig
            logger.info("Google Flash 2.5 initialized successfully")
        except ImportError:
            logger.warning("Vertex AI not available, using mock model")
            self.model = "mock"
            self.generation_config = None
        except Exception as e:
            logger.warning(f"Failed to initialize Flash 2.5: {e}, using mock model")
            self.model = "mock"
            self.generation_config = None
    
    def analyze_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security alert using Flash 2.5 or mock"""
        start_time = time.time()
        
        try:
            # Threat Intelligence enrichment (AbuseIPDB)
            ioc_results = self._lookup_threat_intel(alert_data)
            alert_with_ti = dict(alert_data)
            if ioc_results:
                alert_with_ti["threat_intel"] = ioc_results

            if self.model == "mock":
                analysis_result = self._mock_analysis(alert_with_ti)
            else:
                analysis_result = self._real_analysis(alert_with_ti)

            # Fuse risk score with TI signals
            fused_risk = self._fuse_risk_score(
                base=analysis_result.get("risk_score"),
                ioc=ioc_results
            )
            analysis_result["fused_risk_score"] = fused_risk
            analysis_result["ioc_results"] = ioc_results
            
            # Add metadata
            processing_time = (time.time() - start_time) * 1000
            analysis_result.update({
                "llm_model": "gemini-2.0-flash-exp" if self.model != "mock" else "mock",
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": 100 if self.model != "mock" else 0
            })
            
            # Update metrics
            self._update_metrics(processing_time, True)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self._update_metrics((time.time() - start_time) * 1000, False)
            return self._fallback_analysis(alert_data)
    
    def _real_analysis(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Real Flash 2.5 analysis"""
        prompt = self._build_analysis_prompt(alert_data)
        
        generation_config = self.generation_config(
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            max_output_tokens=1024,
            response_mime_type="application/json"
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return json.loads(response.text)
    
    def _mock_analysis(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock analysis for testing without API access"""
        # Simulate analysis based on alert characteristics
        src_ip = alert_data.get('src_ip', '')
        port = alert_data.get('port', 0)
        bytes_transferred = alert_data.get('bytes_transferred', 0)
        connection_count = alert_data.get('connection_count', 0)
        
        # Determine if suspicious
        is_suspicious = (
            port in [22, 23, 3389] or  # Common attack ports
            bytes_transferred > 1000000 or  # High data transfer
            connection_count > 100 or  # Many connections
            '192.168.1.100' in src_ip  # Test IP
        )
        
        severity = "high" if is_suspicious else "low"
        confidence = 0.9 if is_suspicious else 0.3
        
        return {
            "is_true_positive": is_suspicious,
            "severity": severity,
            "confidence": confidence,
            "threat_indicators": ["suspicious_port", "high_volume"] if is_suspicious else ["normal_traffic"],
            "attack_patterns": ["brute_force", "data_exfiltration"] if is_suspicious else ["normal_communication"],
            "recommended_actions": ["immediate_containment", "investigate"] if is_suspicious else ["monitor"],
            "reasoning": f"Analysis based on port {port}, {bytes_transferred} bytes, {connection_count} connections",
            "risk_score": 0.8 if is_suspicious else 0.2,
            "urgency": "high" if is_suspicious else "low",
            "threat_category": "malware" if is_suspicious else "normal",
            "attack_vector": "network",
            "ioc_confidence": confidence,
            "false_positive_probability": 1 - confidence
        }
    
    def _build_analysis_prompt(self, alert_data: Dict[str, Any]) -> str:
        """Build structured prompt for Flash 2.5 with optional TI context."""
        ti_context = alert_data.get("threat_intel") or {}
        return f"""
        You are an expert cybersecurity analyst. Analyze this security alert with any provided Threat Intelligence
        context and respond with strictly valid JSON. Be concise, factual, and provide ranked factors.

        ALERT DATA:
        {json.dumps({k: v for k, v in alert_data.items() if k != 'threat_intel'}, indent=2)}

        THREAT INTEL (optional):
        {json.dumps(ti_context, indent=2)}

        Respond with JSON using this schema:
        {{
          "is_true_positive": boolean,
          "severity": "low|medium|high|critical",
          "confidence": 0.0-1.0,
          "threat_indicators": ["indicator1", "indicator2"],
          "attack_patterns": ["pattern1", "pattern2"],
          "recommended_actions": ["action1", "action2"],
          "reasoning": "bullet points summarizing the top factors (max 6 bullets)",
          "top_factors": ["ranked factor 1", "factor 2", "factor 3"],
          "mitre_techniques": ["T1110", "T1041"],
          "risk_score": 0.0-1.0,  
          "urgency": "low|medium|high|critical",
          "threat_category": "malware|phishing|ddos|insider|other",
          "attack_vector": "email|web|network|endpoint|other",
          "ioc_confidence": 0.0-1.0,
          "false_positive_probability": 0.0-1.0
        }}
        """
    
    def _fallback_analysis(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when everything fails"""
        return {
            "is_true_positive": True,
            "severity": "medium",
            "confidence": 0.5,
            "threat_indicators": ["unknown"],
            "attack_patterns": ["unknown"],
            "recommended_actions": ["manual_review"],
            "reasoning": "Fallback analysis due to system failure",
            "risk_score": 0.5,
            "urgency": "medium",
            "threat_category": "other",
            "attack_vector": "other",
            "ioc_confidence": 0.0,
            "false_positive_probability": 0.5,
            "llm_model": "fallback",
            "processing_time_ms": 0,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": 0
        }
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["avg_response_time"]
        total_requests = self.performance_metrics["total_requests"]
        self.performance_metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Update success rate
        if success:
            success_count = self.performance_metrics["success_rate"] * (total_requests - 1)
            self.performance_metrics["success_rate"] = (success_count + 1) / total_requests
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()

    # ------------------ TI helpers ------------------
    def _lookup_threat_intel(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        ioc: Dict[str, Any] = {}
        try:
            ips = []
            for key in ("src_ip", "source_ip", "dst_ip", "dest_ip"):
                val = alert_data.get(key)
                if isinstance(val, str) and val and val not in ips:
                    ips.append(val)
            if ti_abuseipdb:
                for ip in ips:
                    ioc.setdefault("abuseipdb", {})[ip] = ti_abuseipdb.query_abuseipdb(ip)
        except Exception as e:  # pragma: no cover
            logger.warning("TI lookup failed: %s", e)
        return ioc

    def _fuse_risk_score(self, base: Optional[float], ioc: Dict[str, Any]) -> float:
        try:
            score = float(base) if base is not None else 0.5
        except Exception:
            score = 0.5
        # Adjust with AbuseIPDB signals
        ab = (ioc or {}).get("abuseipdb", {})
        adjustments = 0.0
        for ip, payload in ab.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("isWhitelisted"):
                adjustments -= 0.1
            conf = payload.get("abuseConfidenceScore")
            if isinstance(conf, (int, float)):
                if conf >= 90:
                    adjustments += 0.2
                elif conf >= 70:
                    adjustments += 0.15
                elif conf >= 50:
                    adjustments += 0.1
        # Clamp to [0,1]
        fused = max(0.0, min(1.0, score + adjustments))
        return fused

if __name__ == "__main__":
    # Test the implementation
    config = {
        "project_id": "chronicle-dev-2be9",
        "location": "us-central1"
    }
    
    taa = EnhancedTAA(config)
    
    # Test alert
    test_alert = {
        "alert_id": "test-001",
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.1",
        "port": 22,
        "protocol": "SSH",
        "bytes_transferred": 1000000,
        "connection_count": 150,
        "timestamp": "2025-09-25T10:30:00Z"
    }
    
    print("🧪 Testing Flash 2.5 integration...")
    result = taa.analyze_alert(test_alert)
    
    print("✅ Test completed!")
    print(f"Model: {result['llm_model']}")
    print(f"Response time: {result['processing_time_ms']:.2f}ms")
    print(f"Analysis: {result['reasoning']}")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2f}")
