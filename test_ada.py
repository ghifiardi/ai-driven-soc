import asyncio
from langgraph_ada_integration import LangGraphAnomalyDetectionAgent, ADAConfig, AlertData

async def main():
    # Create a basic configuration
    config = ADAConfig(
        project_id="test-project",
        location="us-central1",
        redis_url="redis://localhost:6379",  # You'll need Redis running locally
        confidence_threshold=0.8
    )
    
    # Initialize the ADA agent
    ada = LangGraphAnomalyDetectionAgent(config)
    
    # Create a test alert
    test_alert = AlertData(
        log_id="test-log-123",
        timestamp="2024-03-20T12:34:56Z",
        source_ip="192.168.1.100",
        dest_ip="10.1.1.50",
        protocol="tcp",
        port=443,
        bytes_sent=1500,
        bytes_received=8500,
        duration=2.5,
        raw_log={"test": "data"}
    )
    
    # Process the alert
    result = await ada.process_alert(test_alert)
    print("Processing result:", result)

if __name__ == "__main__":
    asyncio.run(main()) 