from langgraph_ada_integration import LangGraphAnomalyDetectionAgent, ADAConfig
import pprint
import random
from datetime import datetime, timedelta

# Generate a large batch of synthetic logs
LABEL_KEYWORDS = [
    ("Confirmed Attack", ["malware", "attack", "exploit", "ransomware", "c2", "command and control"]),
    ("Suspicious Pattern", ["scan", "suspicious", "brute force", "phishing", "recon"]),
    ("Normal Traffic", ["normal", "dns query", "heartbeat", "keepalive"])
]
KILL_CHAIN_MAP = {
    "malware": "Delivery",
    "attack": "Action on Objectives",
    "exploit": "Exploitation",
    "ransomware": "Action on Objectives",
    "c2": "Command and Control",
    "command and control": "Command and Control",
    "scan": "Reconnaissance",
    "suspicious": "Reconnaissance",
    "brute force": "Initial Access",
    "phishing": "Initial Access",
    "recon": "Reconnaissance"
}
USERNAMES = ["alice", "bob", "carol", "dave", "eve", "frank"]
PROTOCOLS = ["tcp", "udp", "icmp"]

random.seed(42)
def random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"

def random_log(idx):
    # Pick a label and keywords
    label, keywords = random.choice(LABEL_KEYWORDS)
    keyword = random.choice(keywords)
    kill_chain = KILL_CHAIN_MAP.get(keyword, "None")
    # Compose log
    return {
        "log_id": f"log{idx}",
        "timestamp": (datetime(2025, 6, 16, 10, 0, 0) + timedelta(seconds=idx*10)).isoformat(),
        "source_ip": random_ip(),
        "dest_ip": random_ip(),
        "protocol": random.choice(PROTOCOLS),
        "port": random.choice([22, 53, 80, 443, 8080]),
        "bytes_sent": random.randint(100, 10000),
        "bytes_received": random.randint(100, 10000),
        "duration": round(random.uniform(0.1, 20.0), 2),
        "username": random.choice(USERNAMES),
        "raw_log": f"User {random.choice(USERNAMES)} {keyword} event detected from {random_ip()}"
    }

# Create 1000 logs, with some duplicates
batch_logs = [random_log(i) for i in range(950)]
batch_logs += [batch_logs[10], batch_logs[20], batch_logs[30], batch_logs[40], batch_logs[50]]  # Add some duplicates

def test_scale_ai_large_batch_preprocessing():
    print("Starting large batch preprocessing test...")
    config = ADAConfig()
    agent = LangGraphAnomalyDetectionAgent(config)
    state = {"alert_data": batch_logs, "batch_id": "test-large-batch"}
    result = agent.scale_ai_preprocessing_node(state)
    logs = result["alert_data"]
    print(f"Total logs after deduplication: {len(logs)}")
    # Print label and kill_chain statistics
    label_counts = {}
    kill_chain_counts = {}
    for log in logs:
        label = log.get("label", "None")
        kill_chain = log.get("kill_chain_stage", "None")
        label_counts[label] = label_counts.get(label, 0) + 1
        kill_chain_counts[kill_chain] = kill_chain_counts.get(kill_chain, 0) + 1
    print("Label distribution:")
    pprint.pprint(label_counts)
    print("Kill chain stage distribution:")
    pprint.pprint(kill_chain_counts)
    # Spot check a few logs
    print("Spot check first 3 logs:")
    for log in logs[:3]:
        pprint.pprint(log)
        assert log["username"] == "user", "username should be standardized to 'user'!"
        assert log["source_ip"].count(".") == 3, "source_ip should look like an IP!"
        assert log["dest_ip"].count(".") == 3, "dest_ip should look like an IP!"
        assert log["label"] in label_counts, "Label should be present!"
        assert log["kill_chain_stage"] in kill_chain_counts, "Kill chain should be present!"
    print("Large batch test completed successfully.")

if __name__ == "__main__":
    test_scale_ai_large_batch_preprocessing()
