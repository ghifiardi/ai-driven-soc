#!/usr/bin/env python3
"""
Embedding Service for AI-Driven SOC
Generates contextual embeddings using Vertex AI for telemetry data
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# Google Cloud imports
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using Vertex AI"""
    
    def __init__(self, project_id: str = "chronicle-dev-2be9", location: str = "us-central1"):
        """Initialize the embedding service"""
        self.project_id = project_id
        self.location = location
        self.model = None
        self._initialize_vertex_ai()
        
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI and embedding model"""
        try:
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Initialize the embedding model
            self.model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            logger.info("Vertex AI embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def generate_embedding(self, text: str, max_tokens: int = 8000) -> Optional[List[float]]:
        """Generate embedding for a single text
        
        Args:
            text: Input text to embed
            max_tokens: Maximum tokens to process (default 8000)
            
        Returns:
            List of 768 float values representing the embedding, or None if failed
        """
        try:
            # Truncate text if too long
            if len(text) > max_tokens * 4:  # Rough character to token ratio
                text = text[:max_tokens * 4]
                logger.warning(f"Text truncated to {max_tokens * 4} characters")
            
            # Generate embedding
            embeddings = self.model.get_embeddings([text])
            
            if embeddings and len(embeddings) > 0:
                return embeddings[0].values
            else:
                logger.error("No embedding generated")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], max_tokens: int = 8000) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts to embed
            max_tokens: Maximum tokens per text
            
        Returns:
            List of embeddings (768 float values each) or None for failed texts
        """
        try:
            # Truncate texts if too long
            truncated_texts = []
            for text in texts:
                if len(text) > max_tokens * 4:
                    truncated_texts.append(text[:max_tokens * 4])
                    logger.warning(f"Text truncated to {max_tokens * 4} characters")
                else:
                    truncated_texts.append(text)
            
            # Generate embeddings in batch
            embeddings = self.model.get_embeddings(truncated_texts)
            
            # Convert to list of lists
            result = []
            for i, embedding in enumerate(embeddings):
                if embedding and embedding.values:
                    result.append(embedding.values)
                else:
                    logger.error(f"No embedding generated for text {i}")
                    result.append(None)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def process_alert_for_embedding(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process alert data to extract text for embedding
        
        Args:
            alert_data: Alert data dictionary
            
        Returns:
            Alert data with embedding added
        """
        try:
            # Extract text from alert data
            text_parts = []
            
            # Add key fields that should be embedded
            if 'raw_log' in alert_data:
                text_parts.append(str(alert_data['raw_log']))
            
            if 'classification' in alert_data:
                text_parts.append(f"Classification: {alert_data['classification']}")
            
            if 'source_ip' in alert_data and 'dest_ip' in alert_data:
                text_parts.append(f"Network: {alert_data['source_ip']} -> {alert_data['dest_ip']}")
            
            if 'bytes_sent' in alert_data and 'bytes_received' in alert_data:
                text_parts.append(f"Traffic: {alert_data['bytes_sent']} sent, {alert_data['bytes_received']} received")
            
            # Combine text parts
            combined_text = " | ".join(text_parts)
            
            # Generate embedding
            embedding = self.generate_embedding(combined_text)
            
            if embedding:
                alert_data['embedding'] = embedding
                alert_data['embedding_timestamp'] = datetime.now().isoformat()
                alert_data['embedding_model'] = "text-embedding-004"
                logger.info(f"Generated embedding for alert {alert_data.get('alert_id', 'unknown')}")
            else:
                logger.warning(f"Failed to generate embedding for alert {alert_data.get('alert_id', 'unknown')}")
            
            return alert_data
            
        except Exception as e:
            logger.error(f"Error processing alert for embedding: {e}")
            return alert_data

# Test function
def test_embedding_service():
    """Test the embedding service with sample data"""
    try:
        # Initialize service
        service = EmbeddingService()
        
        # Test single embedding
        test_text = "Suspicious network activity detected from 192.168.1.100 to 10.0.0.1"
        embedding = service.generate_embedding(test_text)
        
        if embedding:
            print(f"✅ Single embedding generated: {len(embedding)} dimensions")
        else:
            print("❌ Failed to generate single embedding")
        
        # Test batch embeddings
        test_texts = [
            "Normal user login from 192.168.1.50",
            "High volume data transfer detected",
            "Failed authentication attempt from 10.0.0.5"
        ]
        
        embeddings = service.generate_embeddings_batch(test_texts)
        successful_embeddings = [e for e in embeddings if e is not None]
        
        print(f"✅ Batch embeddings: {len(successful_embeddings)}/{len(test_texts)} successful")
        
        # Test similarity calculation
        if len(successful_embeddings) >= 2:
            similarity = service.calculate_similarity(embeddings[0], embeddings[1])
            print(f"✅ Similarity calculation: {similarity:.4f}")
        
        # Test alert processing
        sample_alert = {
            'alert_id': 'test_001',
            'raw_log': '{"timestamp": "2024-01-01T10:00:00Z", "event": "login", "user": "admin"}',
            'classification': 'suspicious',
            'source_ip': '192.168.1.100',
            'dest_ip': '10.0.0.1',
            'bytes_sent': 1024,
            'bytes_received': 2048
        }
        
        processed_alert = service.process_alert_for_embedding(sample_alert)
        
        if 'embedding' in processed_alert:
            print(f"✅ Alert processing: Embedding added with {len(processed_alert['embedding'])} dimensions")
        else:
            print("❌ Failed to process alert for embedding")
        
        print("🎉 Embedding service test completed successfully!")
        
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")

if __name__ == "__main__":
    test_embedding_service()
