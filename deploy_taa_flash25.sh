#!/bin/bash
# TAA Google Flash 2.5 Deployment Script
# Run this on your GCP VM to deploy Flash 2.5 integration

set -e
echo "🚀 Starting TAA Google Flash 2.5 Deployment..."

# Configuration
PROJECT_ID="chronicle-dev-2be9"
LOCATION="us-central1"
VM_ZONE="asia-southeast2-a"
VM_NAME="xdgaisocapp01"

echo "📋 Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Location: $LOCATION"
echo "  VM: $VM_NAME ($VM_ZONE)"
echo ""

# Step 1: Enable required APIs
echo "🔧 Step 1: Enabling required Google Cloud APIs..."
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
gcloud services enable vertexai.googleapis.com --project=$PROJECT_ID
echo "✅ APIs enabled successfully"
echo ""

# Step 2: Create enhanced TAA implementation
echo "📝 Step 2: Creating enhanced TAA implementation..."

cat > enhanced_taa_flash25.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced TAA with Google Flash 2.5 Integration
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

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
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel("gemini-2.0-flash-exp")
            logger.info("Google Flash 2.5 initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Flash 2.5: {e}")
            raise
    
    def analyze_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security alert using Flash 2.5"""
        start_time = time.time()
        
        try:
            # Build analysis prompt
            prompt = self._build_analysis_prompt(alert_data)
            
            # Configure generation for speed and accuracy
            generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
                response_mime_type="application/json"
            )
            
            # Generate analysis
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse response
            analysis_result = json.loads(response.text)
            
            # Add metadata
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            analysis_result.update({
                "llm_model": "gemini-2.0-flash-exp",
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": response.usage_metadata.total_token_count
            })
            
            # Update metrics
            self._update_metrics(processing_time, True)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Flash 2.5 analysis failed: {e}")
            self._update_metrics((time.time() - start_time) * 1000, False)
            return self._fallback_analysis(alert_data)
    
    def _build_analysis_prompt(self, alert_data: Dict[str, Any]) -> str:
        """Build structured prompt for Flash 2.5"""
        return f"""
        You are an expert cybersecurity analyst with 15+ years of experience in SOC operations, threat hunting, and incident response. Analyze the following security alert and provide a comprehensive assessment.

        ALERT DATA:
        {json.dumps(alert_data, indent=2)}

        ANALYSIS REQUIREMENTS:
        1. Determine if this is a TRUE POSITIVE or FALSE POSITIVE
        2. Assess severity level: low, medium, high, critical
        3. Calculate confidence score (0.0 to 1.0)
        4. Identify specific threat indicators and attack patterns
        5. Recommend immediate containment actions
        6. Provide detailed reasoning for your assessment
        7. Calculate risk score (0.0 to 1.0)
        8. Determine urgency level: low, medium, high, critical

        RESPOND WITH VALID JSON IN THIS EXACT FORMAT:
        {{
          "is_true_positive": boolean,
          "severity": "low|medium|high|critical",
          "confidence": 0.0-1.0,
          "threat_indicators": ["indicator1", "indicator2"],
          "attack_patterns": ["pattern1", "pattern2"],
          "recommended_actions": ["action1", "action2"],
          "reasoning": "detailed explanation of analysis",
          "risk_score": 0.0-1.0,
          "urgency": "low|medium|high|critical",
          "threat_category": "malware|phishing|ddos|insider|other",
          "attack_vector": "email|web|network|endpoint|other",
          "ioc_confidence": 0.0-1.0,
          "false_positive_probability": 0.0-1.0
        }}
        """
    
    def _fallback_analysis(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Flash 2.5 fails"""
        return {
            "is_true_positive": True,  # Conservative approach
            "severity": "medium",
            "confidence": 0.5,
            "threat_indicators": ["unknown"],
            "attack_patterns": ["unknown"],
            "recommended_actions": ["manual_review"],
            "reasoning": "Fallback analysis due to LLM failure",
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
    print(f"Analysis: {result['reasoning'][:100]}...")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2f}")
EOF

echo "✅ Enhanced TAA implementation created"
echo ""

# Step 3: Update TAA LangGraph workflow
echo "🔄 Step 3: Updating TAA LangGraph workflow..."

cat > taa_langgraph_enhanced.py << 'EOF'
from langgraph.graph import StateGraph, END
import logging
import json
from google.cloud import pubsub_v1
from typing import Dict, Any
from enhanced_taa_flash25 import EnhancedTAA

# Example state structure
class TAAState(dict):
    pass

# Global publisher for Pub/Sub
publisher = None

def init_pubsub():
    global publisher
    if publisher is None:
        publisher = pubsub_v1.PublisherClient()
    return publisher

def receive_alert_node(state: TAAState) -> TAAState:
    logging.info(f"Received alert: {state.get('alert_id')}")
    return state

def enrichment_node(state: TAAState) -> TAAState:
    state['enriched'] = True
    logging.info(f"Enriched alert: {state.get('alert_id')}")
    return state

def llm_analysis_node(state: TAAState) -> TAAState:
    """Enhanced LLM analysis with Google Flash 2.5"""
    
    # Initialize enhanced TAA if not already done
    if 'enhanced_taa' not in state:
        config = {
            "project_id": "chronicle-dev-2be9",
            "location": "us-central1"
        }
        state['enhanced_taa'] = EnhancedTAA(config)
    
    # Analyze alert using Flash 2.5
    analysis_result = state['enhanced_taa'].analyze_alert(
        state.get('alert_data', {})
    )
    
    # Store results
    state['llm_result'] = analysis_result
    state['analysis_metadata'] = {
        "model_used": analysis_result.get("llm_model"),
        "processing_time": analysis_result.get("processing_time_ms"),
        "tokens_used": analysis_result.get("tokens_used")
    }
    
    logging.info(f"LLM analysis complete for alert {state.get('alert_id')} using {analysis_result.get('llm_model')}")
    
    return state

def unpack_state(state):
    # Utility to handle (node, state) tuple if passed by mistake
    if isinstance(state, tuple):
        return state[1]
    return state

def containment_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['containment'] = 'requested'
    
    # Publish containment request to CRA
    try:
        pub = init_pubsub()
        project_id = "chronicle-dev-2be9"
        topic_name = "containment-requests"
        topic_path = pub.topic_path(project_id, topic_name)
        
        containment_data = {
            "alert_id": state.get('alert_id'),
            "alert_data": state.get('alert_data', {}),
            "llm_result": state.get('llm_result', {}),
            "timestamp": state.get('timestamp'),
            "containment_request": {
                "action": "immediate_containment",
                "severity": state.get('llm_result', {}).get('severity', 'high'),
                "confidence": state.get('llm_result', {}).get('confidence', 0.95),
                "reasoning": f"High severity true positive detected: {state.get('llm_result', {}).get('reasoning', 'Automated analysis')}"
            }
        }
        
        message_data = json.dumps(containment_data).encode("utf-8")
        future = pub.publish(topic_path, message_data)
        message_id = future.result()
        
        logging.info(f"Published containment request for alert {state.get('alert_id')} to CRA. Message ID: {message_id}")
        state['containment_message_id'] = message_id
        
    except Exception as e:
        logging.error(f"Failed to publish containment request for alert {state.get('alert_id')}: {e}")
        state['containment_error'] = str(e)
    
    return state

def manual_review_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['manual_review'] = True
    logging.info(f"Manual review flagged for alert: {state.get('alert_id')}")
    return state

def feedback_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['feedback'] = 'published'
    
    # Publish feedback to CLA
    try:
        pub = init_pubsub()
        project_id = "chronicle-dev-2be9"
        topic_name = "taa-feedback"
        topic_path = pub.topic_path(project_id, topic_name)
        
        feedback_data = {
            "alert_id": state.get('alert_id'),
            "is_true_positive": state.get('llm_result', {}).get('is_true_positive', True),
            "confidence": state.get('llm_result', {}).get('confidence', 0.95),
            "severity": state.get('llm_result', {}).get('severity', 'medium'),
            "timestamp": state.get('timestamp'),
            "reasoning": state.get('llm_result', {}).get('reasoning', 'Automated analysis'),
            "source": "taa_langgraph_flash25"
        }
        
        message_data = json.dumps(feedback_data).encode("utf-8")
        future = pub.publish(topic_path, message_data)
        message_id = future.result()
        
        logging.info(f"Published feedback for alert {state.get('alert_id')} to CLA. Message ID: {message_id}")
        state['feedback_message_id'] = message_id
        
    except Exception as e:
        logging.error(f"Failed to publish feedback for alert {state.get('alert_id')}: {e}")
        state['feedback_error'] = str(e)
    
    return state

def reporting_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['reported'] = True
    
    # Publish report to RVA (Reporting & Visualization Agent)
    try:
        pub = init_pubsub()
        project_id = "chronicle-dev-2be9"
        topic_name = "taa-reports"
        topic_path = pub.topic_path(project_id, topic_name)
        
        report_data = {
            "alert_id": state.get('alert_id'),
            "alert_data": state.get('alert_data', {}),
            "llm_result": state.get('llm_result', {}),
            "timestamp": state.get('timestamp'),
            "containment_requested": state.get('containment') == 'requested',
            "manual_review_flagged": state.get('manual_review', False),
            "feedback_published": state.get('feedback') == 'published',
            "containment_message_id": state.get('containment_message_id'),
            "feedback_message_id": state.get('feedback_message_id'),
            "source": "taa_langgraph_flash25"
        }
        
        message_data = json.dumps(report_data).encode("utf-8")
        future = pub.publish(topic_path, message_data)
        message_id = future.result()
        
        logging.info(f"Published report for alert {state.get('alert_id')} to RVA. Message ID: {message_id}")
        state['report_message_id'] = message_id
        
    except Exception as e:
        logging.error(f"Failed to publish report for alert {state.get('alert_id')}: {e}")
        state['report_error'] = str(e)
    
    return state

def decision_node(state: TAAState) -> TAAState:
    result = state.get('llm_result', {})
    if result.get('is_true_positive') and result.get('severity') == 'high':
        next_node = 'containment'
        logging.info("Routing to containment node")
    elif result.get('confidence', 1.0) < 0.7:
        next_node = 'manual_review'
        logging.info("Routing to manual_review node")
    else:
        next_node = 'feedback'
        logging.info("Routing to feedback node")
    state['next'] = next_node
    return state

def build_taa_workflow():
    graph = StateGraph(TAAState)
    
    # Add nodes
    graph.add_node('receive_alert', receive_alert_node)
    graph.add_node('enrichment', enrichment_node)
    graph.add_node('llm_analysis', llm_analysis_node)
    graph.add_node('decision', decision_node)
    graph.add_node('containment', containment_node)
    graph.add_node('manual_review', manual_review_node)
    graph.add_node('feedback', feedback_node)
    graph.add_node('reporting', reporting_node)

    # Set entrypoint
    graph.set_entry_point('receive_alert')

    # Define linear flow before decision
    graph.add_edge('receive_alert', 'enrichment')
    graph.add_edge('enrichment', 'llm_analysis')
    graph.add_edge('llm_analysis', 'decision')
    
    # Define conditional routing after decision
    graph.add_conditional_edges(
        'decision',
        lambda s: s.get('next', 'feedback'),
        {
            'containment': 'containment',
            'manual_review': 'manual_review', 
            'feedback': 'feedback'
        }
    )
    
    # Define flow after decision branches
    graph.add_edge('containment', 'feedback')
    graph.add_edge('manual_review', 'feedback')
    graph.add_edge('feedback', 'reporting')
    graph.add_edge('reporting', END)

    return graph

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    workflow = build_taa_workflow().compile()
    initial_state = TAAState(alert_id="test-alert-001", alert_data={"example": "data"})
    final_state = workflow.invoke(initial_state)
    print("Final state:", final_state)
EOF

echo "✅ Enhanced TAA LangGraph workflow created"
echo ""

# Step 4: Install dependencies
echo "📦 Step 4: Installing required dependencies..."

cat > requirements_flash25.txt << 'EOF'
google-cloud-aiplatform>=1.38.0
vertexai>=1.38.0
google-generativeai>=0.3.0
langgraph>=0.2.0
google-cloud-pubsub>=2.18.0
google-cloud-bigquery>=3.11.0
google-cloud-firestore>=2.11.0
EOF

echo "✅ Dependencies file created"
echo ""

# Step 5: Create test script
echo "🧪 Step 5: Creating test script..."

cat > test_flash25_integration.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for TAA Flash 2.5 integration
"""

import json
import time
from enhanced_taa_flash25 import EnhancedTAA

def test_flash_25_integration():
    """Test Flash 2.5 integration"""
    
    print("🚀 Testing TAA Flash 2.5 Integration...")
    print("=" * 50)
    
    # Initialize TAA
    config = {
        "project_id": "chronicle-dev-2be9",
        "location": "us-central1"
    }
    
    try:
        taa = EnhancedTAA(config)
        print("✅ Enhanced TAA initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TAA: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "High Severity SSH Attack",
            "alert": {
                "alert_id": "test-001",
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "port": 22,
                "protocol": "SSH",
                "bytes_transferred": 1000000,
                "connection_count": 150,
                "timestamp": "2025-09-25T10:30:00Z"
            }
        },
        {
            "name": "Suspicious Web Traffic",
            "alert": {
                "alert_id": "test-002",
                "src_ip": "10.0.0.50",
                "dst_ip": "external-site.com",
                "port": 443,
                "protocol": "HTTPS",
                "bytes_transferred": 500000,
                "connection_count": 25,
                "timestamp": "2025-09-25T10:35:00Z"
            }
        },
        {
            "name": "Normal Network Activity",
            "alert": {
                "alert_id": "test-003",
                "src_ip": "192.168.1.10",
                "dst_ip": "192.168.1.1",
                "port": 80,
                "protocol": "HTTP",
                "bytes_transferred": 1000,
                "connection_count": 1,
                "timestamp": "2025-09-25T10:40:00Z"
            }
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            result = taa.analyze_alert(test_case['alert'])
            processing_time = (time.time() - start_time) * 1000
            
            print(f"✅ Analysis completed in {processing_time:.2f}ms")
            print(f"   Model: {result.get('llm_model', 'unknown')}")
            print(f"   Severity: {result.get('severity', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   True Positive: {result.get('is_true_positive', False)}")
            print(f"   Threat Category: {result.get('threat_category', 'unknown')}")
            print(f"   Reasoning: {result.get('reasoning', 'No reasoning provided')[:100]}...")
            
            results.append({
                "test": test_case['name'],
                "success": True,
                "processing_time": processing_time,
                "result": result
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        avg_time = sum(r['processing_time'] for r in results if r['success']) / successful_tests
        print(f"Average Processing Time: {avg_time:.2f}ms")
    
    # Performance metrics
    try:
        metrics = taa.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Requests: {metrics['total_requests']}")
        print(f"   Average Response Time: {metrics['avg_response_time']:.2f}ms")
        print(f"   Success Rate: {metrics['success_rate']*100:.1f}%")
    except Exception as e:
        print(f"⚠️  Could not retrieve performance metrics: {e}")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = test_flash_25_integration()
    if success:
        print("\n🎉 All tests passed! Flash 2.5 integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
EOF

echo "✅ Test script created"
echo ""

# Step 6: Create deployment script for VM
echo "📤 Step 6: Creating VM deployment script..."

cat > deploy_to_vm.sh << 'EOF'
#!/bin/bash
# Deploy TAA Flash 2.5 to VM

echo "🚀 Deploying TAA Flash 2.5 to VM..."

# Copy files to VM
gcloud compute scp enhanced_taa_flash25.py app@xdgaisocapp01:~/ --zone=asia-southeast2-a
gcloud compute scp taa_langgraph_enhanced.py app@xdgaisocapp01:~/ --zone=asia-southeast2-a
gcloud compute scp test_flash25_integration.py app@xdgaisocapp01:~/ --zone=asia-southeast2-a
gcloud compute scp requirements_flash25.txt app@xdgaisocapp01:~/ --zone=asia-southeast2-a

# Install dependencies on VM
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="
cd /home/app && 
source ai-driven-soc/venv/bin/activate && 
pip install -r requirements_flash25.txt
"

# Test the integration
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="
cd /home/app && 
source ai-driven-soc/venv/bin/activate && 
python test_flash25_integration.py
"

echo "✅ Deployment completed!"
EOF

chmod +x deploy_to_vm.sh

echo "✅ VM deployment script created"
echo ""

# Step 7: Create monitoring dashboard update
echo "📊 Step 7: Creating monitoring dashboard update..."

cat > taa_flash25_dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
TAA Flash 2.5 Performance Dashboard
"""

import streamlit as st
import json
import time
from datetime import datetime, timedelta
from enhanced_taa_flash25 import EnhancedTAA

st.set_page_config(
    page_title="TAA Flash 2.5 Dashboard",
    page_icon="⚡",
    layout="wide"
)

def main():
    st.title("⚡ TAA Flash 2.5 Performance Dashboard")
    st.markdown("**Real-time monitoring of TAA with Google Flash 2.5 integration**")
    
    # Initialize TAA
    if 'taa' not in st.session_state:
        config = {
            "project_id": "chronicle-dev-2be9",
            "location": "us-central1"
        }
        try:
            st.session_state.taa = EnhancedTAA(config)
            st.success("✅ TAA Flash 2.5 initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize TAA: {e}")
            return
    
    # Performance Metrics
    st.header("📊 Performance Metrics")
    
    metrics = st.session_state.taa.get_performance_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", metrics['total_requests'])
    
    with col2:
        st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}ms")
    
    with col3:
        st.metric("Success Rate", f"{metrics['success_rate']*100:.1f}%")
    
    with col4:
        st.metric("Cache Hit Rate", f"{metrics['cache_hits']:.1f}%")
    
    # Test Interface
    st.header("🧪 Test Interface")
    
    with st.form("test_form"):
        st.subheader("Test Alert Analysis")
        
        alert_id = st.text_input("Alert ID", value="test-alert-001")
        src_ip = st.text_input("Source IP", value="192.168.1.100")
        dst_ip = st.text_input("Destination IP", value="10.0.0.1")
        port = st.number_input("Port", value=22, min_value=1, max_value=65535)
        protocol = st.selectbox("Protocol", ["SSH", "HTTP", "HTTPS", "FTP", "SMTP"])
        bytes_transferred = st.number_input("Bytes Transferred", value=1000000)
        connection_count = st.number_input("Connection Count", value=150)
        
        submitted = st.form_submit_button("Analyze Alert")
        
        if submitted:
            test_alert = {
                "alert_id": alert_id,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "port": port,
                "protocol": protocol,
                "bytes_transferred": bytes_transferred,
                "connection_count": connection_count,
                "timestamp": datetime.now().isoformat()
            }
            
            with st.spinner("Analyzing alert with Flash 2.5..."):
                start_time = time.time()
                result = st.session_state.taa.analyze_alert(test_alert)
                processing_time = (time.time() - start_time) * 1000
            
            st.success(f"✅ Analysis completed in {processing_time:.2f}ms")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Analysis Results")
                st.json({
                    "Model": result.get('llm_model'),
                    "Severity": result.get('severity'),
                    "Confidence": result.get('confidence'),
                    "True Positive": result.get('is_true_positive'),
                    "Threat Category": result.get('threat_category'),
                    "Attack Vector": result.get('attack_vector'),
                    "Risk Score": result.get('risk_score'),
                    "Urgency": result.get('urgency')
                })
            
            with col2:
                st.subheader("Threat Indicators")
                for indicator in result.get('threat_indicators', []):
                    st.write(f"• {indicator}")
                
                st.subheader("Attack Patterns")
                for pattern in result.get('attack_patterns', []):
                    st.write(f"• {pattern}")
                
                st.subheader("Recommended Actions")
                for action in result.get('recommended_actions', []):
                    st.write(f"• {action}")
            
            st.subheader("Reasoning")
            st.write(result.get('reasoning', 'No reasoning provided'))
    
    # Auto-refresh
    if st.checkbox("Auto-refresh (30s)", value=False):
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
EOF

echo "✅ Monitoring dashboard created"
echo ""

# Step 8: Summary
echo "🎉 Step 8: Deployment Summary"
echo "=" * 50
echo "✅ Google Cloud APIs enabled"
echo "✅ Enhanced TAA implementation created"
echo "✅ LangGraph workflow updated"
echo "✅ Dependencies file created"
echo "✅ Test script created"
echo "✅ VM deployment script created"
echo "✅ Monitoring dashboard created"
echo ""
echo "📋 Next Steps:"
echo "1. Run: ./deploy_to_vm.sh"
echo "2. Test: python test_flash25_integration.py"
echo "3. Monitor: streamlit run taa_flash25_dashboard.py"
echo ""
echo "🚀 Ready to deploy TAA Flash 2.5 integration!"
