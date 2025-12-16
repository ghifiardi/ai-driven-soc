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
