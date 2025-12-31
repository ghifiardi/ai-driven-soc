#!/usr/bin/env python3
"""
GATRA Anomaly Detection System
==============================
Advanced ensemble-based anomaly detection for AI-driven SOC.
Incorporates Neural Autoencoders, Graph Analysis, and Causal Inference.
"""

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float  # 0 to 1
    severity: SeverityLevel
    detector_type: str
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# --- 1. Autoencoder-Based Detection ---

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super(AutoencoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderDetector:
    def __init__(self, input_dim: int, threshold: float = 0.1):
        self.model = AutoencoderModel(input_dim)
        self.threshold = threshold
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.is_trained = False

    def fit(self, data: np.ndarray, epochs: int = 50):
        self.model.train()
        tensor_data = torch.FloatTensor(data)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(tensor_data)
            loss = self.criterion(output, tensor_data)
            loss.backward()
            self.optimizer.step()
        self.is_trained = True
        logger.info(f"Autoencoder trained. Final loss: {loss.item():.6f}")

    def predict_anomaly(self, data: np.ndarray) -> Tuple[bool, float, float]:
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(data)
            output = self.model(tensor_data)
            reconstruction_error = torch.mean((output - tensor_data) ** 2, dim=1).numpy()
            
            # Normalize error to a 0-1 score based on threshold
            score = np.clip(reconstruction_error / (self.threshold * 2), 0, 1)
            is_anomaly = reconstruction_error > self.threshold
            return is_anomaly[0], score[0], reconstruction_error[0]

# --- 2. Graph-Based Detection ---

class GraphAnomalyDetector:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.baseline_stats = {}

    def add_event(self, source: str, destination: str, weight: float = 1.0):
        if self.graph.has_edge(source, destination):
            self.graph[source][destination]['weight'] += weight
        else:
            self.graph.add_edge(source, destination, weight=weight)

    def detect_structural_anomaly(self, source: str, destination: str) -> Tuple[float, List[str]]:
        reasons = []
        score = 0.0
        
        # 1. Check if edge is new
        if not self.graph.has_edge(source, destination):
            score += 0.4
            reasons.append(f"Unexpected new connection from {source} to {destination}")
        
        # 2. Check node degree (centrality change)
        if source in self.graph:
            degree = self.graph.degree(source)
            if degree > 100:  # Arbitrary threshold for demo
                score += 0.3
                reasons.append(f"Source node {source} has unusually high degree: {degree}")
        
        # 3. Check for cycles (potential DNS tunneling or C2)
        try:
            if nx.has_path(self.graph, destination, source):
                score += 0.5
                reasons.append(f"Circular dependency detected between {source} and {destination}")
        except nx.NodeNotFound:
            pass

        return min(score, 1.0), reasons

# --- 3. Causal Inference Engine ---

class CausalInferenceEngine:
    def __init__(self):
        self.event_log = []

    def add_event(self, event_type: str, timestamp: datetime, metadata: Dict[str, Any]):
        self.event_log.append({
            'type': event_type,
            'timestamp': timestamp,
            'metadata': metadata
        })

    def find_root_cause(self, target_event: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified causal analysis based on temporal proximity"""
        target_time = target_event.get('timestamp', datetime.now())
        if isinstance(target_time, str):
            if target_time == 'now':
                target_time = datetime.now()
            else:
                try:
                    target_time = datetime.fromisoformat(target_time)
                except ValueError:
                    target_time = datetime.now()
            
        # Look for events within 60 seconds before
        potential_causes = []
        for event in self.event_log:
            time_diff = (target_time - event['timestamp']).total_seconds()
            if 0 < time_diff < 60:
                potential_causes.append(event)
        
        return {
            'target': target_event.get('type', 'Unknown'),
            'root_causes': [e['type'] for e in potential_causes],
            'confidence': 0.7 if potential_causes else 0.1
        }

# --- 4. Main System (Ensemble) ---

class GATRAAnomalyDetectionSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ae_detector = AutoencoderDetector(input_dim=10) # Example dim
        self.graph_detector = GraphAnomalyDetector()
        self.causal_engine = CausalInferenceEngine()
        
    def process_telemetry(self, telemetry: Dict[str, Any]) -> AnomalyResult:
        """Process incoming telemetry through the ensemble"""
        source = telemetry.get('source_ip', 'unknown')
        dest = telemetry.get('dest_ip', 'unknown')
        features = telemetry.get('features', np.zeros(10))
        
        # 1. AE Detection
        is_ae_anomaly, ae_score, _ = self.ae_detector.predict_anomaly(features.reshape(1, -1))
        
        # 2. Graph Detection
        graph_score, graph_reasons = self.graph_detector.detect_structural_anomaly(source, dest)
        self.graph_detector.add_event(source, dest)
        
        # 3. Causal Analysis (if anomaly detected)
        root_cause = {}
        if is_ae_anomaly or graph_score > 0.5:
            root_cause = self.causal_engine.find_root_cause(telemetry)
            self.causal_engine.add_event(
                telemetry.get('event_type', 'alert'),
                datetime.now(),
                telemetry
            )
            
        # Combine Scores (weighted average)
        combined_score = float((ae_score * 0.6) + (graph_score * 0.4))
        is_anomaly = bool(combined_score > 0.5)
        
        severity = SeverityLevel.INFO
        if combined_score > 0.8: severity = SeverityLevel.CRITICAL
        elif combined_score > 0.6: severity = SeverityLevel.HIGH
        elif combined_score > 0.4: severity = SeverityLevel.MEDIUM
        elif combined_score > 0.2: severity = SeverityLevel.LOW
        
        reasons = graph_reasons
        if is_ae_anomaly:
            reasons.append(f"Autoencoder reconstruction error above threshold (Score: {ae_score:.2f})")
        if root_cause.get('root_causes'):
            reasons.append(f"Potential root causes: {', '.join(root_cause['root_causes'])}")
            
        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=combined_score,
            severity=severity,
            detector_type="GATRA_Ensemble",
            reasoning=reasons,
            metadata={
                'ae_score': float(ae_score),
                'graph_score': float(graph_score),
                'root_cause': root_cause
            }
        )

if __name__ == "__main__":
    # Demo
    system = GATRAAnomalyDetectionSystem()
    
    # Train AE with dummy data
    normal_data = np.random.rand(100, 10)
    system.ae_detector.fit(normal_data)
    
    # Test one alert
    test_alert = {
        'source_ip': '1.2.3.4',
        'dest_ip': '8.8.8.8',
        'features': np.random.rand(10),
        'timestamp': datetime.now().isoformat(),
        'event_type': 'outbound_connection'
    }
    
    result = system.process_telemetry(test_alert)
    print(f"Result: {result}")
