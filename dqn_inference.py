"""
Production DQN Inference Module for Cyber Defense
Extracted from cyber_defense_agent.py for production use
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """Deep Q-Network for cyber defense decisions"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ProductionDQNAgent:
    """Production DQN Agent for real-time cyber defense decisions"""
    
    def __init__(self, model_path: Optional[str] = None, state_size: int = 7, action_size: int = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Action mappings for explainability
        self.action_names = {
            0: "monitor",
            1: "block_traffic", 
            2: "patch_system",
            3: "isolate_node",
            4: "no_action"
        }
        
        # Initialize model
        self.model = DQN(state_size, action_size).to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("No pre-trained model found. Using initialized weights.")
            
        self.model.eval()  # Set to evaluation mode
        
    def load_model(self, model_path: str):
        """Load pre-trained DQN model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Model loaded from {model_path}")
            else:
                # Fallback for direct state dict save
                self.model.load_state_dict(checkpoint)
                logger.info(f"Model state dict loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def save_model(self, model_path: str):
        """Save current model state"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'state_size': self.state_size,
                'action_size': self.action_size
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {model_path}: {e}")
            raise
    
    def alert_to_state(self, alert_data: dict) -> np.ndarray:
        """Convert security alert to DQN state representation"""
        try:
            # Extract relevant features from alert
            # This mapping depends on your alert schema
            num_nodes = self.state_size - 2  # Last 2 are attack_type and threat_level
            
            # Initialize node statuses (default secure)
            node_statuses = np.zeros(num_nodes)
            
            # Map alert fields to state
            attack_type_map = {
                'none': 0, 'malware': 1, 'ddos': 2, 'intrusion': 3,
                'phishing': 1, 'ransomware': 1, 'botnet': 3
            }
            
            # Extract attack type
            attack_type = alert_data.get('attack_type', 'none').lower()
            attack_type_value = attack_type_map.get(attack_type, 0)
            
            # Extract threat level (0-10 scale)
            threat_level = min(10, max(0, alert_data.get('severity', 0)))
            if isinstance(threat_level, str):
                threat_map = {'low': 2, 'medium': 5, 'high': 8, 'critical': 10}
                threat_level = threat_map.get(threat_level.lower(), 0)
            
            # Extract affected nodes/assets
            affected_assets = alert_data.get('affected_assets', [])
            for i, asset in enumerate(affected_assets[:num_nodes]):
                if asset.get('status') == 'compromised':
                    node_statuses[i] = 2
                elif asset.get('status') == 'vulnerable':
                    node_statuses[i] = 1
            
            # Combine into state vector
            state = np.concatenate([
                node_statuses,
                [attack_type_value, threat_level]
            ]).astype(np.float32)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to convert alert to state: {e}")
            # Return safe default state
            return np.zeros(self.state_size).astype(np.float32)
    
    def act(self, state: np.ndarray) -> Tuple[int, float, dict]:
        """
        Make defensive action decision
        
        Returns:
            action (int): Recommended action (0-4)
            confidence (float): Q-value confidence score
            context (dict): Additional decision context
        """
        try:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get Q-values
            with torch.no_grad():
                q_values = self.model(state_tensor)
            
            # Get best action and confidence
            action = q_values.argmax().item()
            confidence = torch.max(q_values).item()
            
            # Create decision context
            q_vals_np = q_values.cpu().numpy().flatten()
            context = {
                'q_values': {
                    self.action_names[i]: float(q_vals_np[i]) 
                    for i in range(len(q_vals_np))
                },
                'action_name': self.action_names[action],
                'confidence_score': float(confidence),
                'state_analysis': self._analyze_state(state)
            }
            
            logger.info(f"DQN Decision: {self.action_names[action]} (confidence: {confidence:.2f})")
            
            return action, confidence, context
            
        except Exception as e:
            logger.error(f"DQN inference failed: {e}")
            # Fallback to safe action
            return 4, 0.0, {'error': str(e), 'fallback': True}
    
    def _analyze_state(self, state: np.ndarray) -> dict:
        """Analyze current state for context"""
        try:
            num_nodes = self.state_size - 2
            node_statuses = state[:num_nodes]
            attack_type = int(state[-2])
            threat_level = int(state[-1])
            
            attack_types = ['none', 'malware', 'ddos', 'intrusion']
            attack_name = attack_types[attack_type] if attack_type < len(attack_types) else 'unknown'
            
            return {
                'total_nodes': int(num_nodes),
                'secure_nodes': int(np.sum(node_statuses == 0)),
                'vulnerable_nodes': int(np.sum(node_statuses == 1)),
                'compromised_nodes': int(np.sum(node_statuses == 2)),
                'attack_type': attack_name,
                'threat_level': int(threat_level)
            }
        except Exception:
            return {'analysis_failed': True}
    
    def get_action_recommendation(self, alert_data: dict) -> dict:
        """
        High-level interface for getting action recommendations
        
        Args:
            alert_data: Security alert dictionary
            
        Returns:
            Dictionary with action, confidence, and reasoning
        """
        # Convert alert to state
        state = self.alert_to_state(alert_data)
        
        # Get DQN decision
        action, confidence, context = self.act(state)
        
        return {
            'recommended_action': self.action_names[action],
            'action_id': action,
            'confidence': confidence,
            'dqn_context': context,
            'alert_state': state.tolist()
        }


class RuleBasedFallback:
    """Fallback decision engine when DQN is unavailable"""
    
    def __init__(self):
        self.action_names = {
            0: "monitor",
            1: "block_traffic", 
            2: "patch_system", 
            3: "isolate_node",
            4: "no_action"
        }
    
    def get_action_recommendation(self, alert_data: dict) -> dict:
        """Rule-based fallback decision making"""
        try:
            # Simple rule-based logic
            severity = alert_data.get('severity', 'low')
            attack_type = alert_data.get('attack_type', '').lower()
            affected_assets = alert_data.get('affected_assets', [])
            
            # Check for compromised assets
            has_compromised = any(
                asset.get('status') == 'compromised' 
                for asset in affected_assets
            )
            
            if has_compromised:
                action = 3  # isolate_node
                confidence = 0.8
                reasoning = "Compromised assets detected, isolation required"
            elif attack_type in ['ddos', 'dos']:
                action = 1  # block_traffic
                confidence = 0.7
                reasoning = f"DDoS attack detected, blocking traffic"
            elif attack_type in ['malware', 'ransomware', 'intrusion']:
                action = 2  # patch_system
                confidence = 0.7
                reasoning = f"{attack_type} detected, patching recommended"
            elif severity in ['high', 'critical']:
                action = 0  # monitor
                confidence = 0.6
                reasoning = "High severity alert, enhanced monitoring"
            else:
                action = 4  # no_action
                confidence = 0.5
                reasoning = "Low priority alert, no immediate action"
            
            return {
                'recommended_action': self.action_names[action],
                'action_id': action,
                'confidence': confidence,
                'fallback_reasoning': reasoning,
                'rule_based': True
            }
            
        except Exception as e:
            return {
                'recommended_action': 'no_action',
                'action_id': 4,
                'confidence': 0.1,
                'error': str(e),
                'rule_based': True
            }


# Factory function for easy instantiation
def create_defense_agent(model_path: Optional[str] = None, 
                        enable_fallback: bool = True) -> ProductionDQNAgent:
    """
    Factory function to create production DQN agent
    
    Args:
        model_path: Path to pre-trained model file
        enable_fallback: Whether to enable rule-based fallback
        
    Returns:
        ProductionDQNAgent instance
    """
    try:
        agent = ProductionDQNAgent(model_path=model_path)
        if enable_fallback:
            agent.fallback = RuleBasedFallback()
        return agent
    except Exception as e:
        logger.error(f"Failed to create DQN agent: {e}")
        if enable_fallback:
            logger.info("Using rule-based fallback only")
            return RuleBasedFallback()
        else:
            raise


if __name__ == "__main__":
    # Quick test of the module
    logging.basicConfig(level=logging.INFO)
    
    # Create agent
    agent = create_defense_agent()
    
    # Test with sample alert
    sample_alert = {
        'attack_type': 'malware',
        'severity': 'high',
        'affected_assets': [
            {'id': 'server1', 'status': 'vulnerable'},
            {'id': 'server2', 'status': 'secure'}
        ]
    }
    
    result = agent.get_action_recommendation(sample_alert)
    print("DQN Inference Test:")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Test successful!")


















