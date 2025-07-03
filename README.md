
# LLM
# AI-Driven SOC: LangGraph ADA Integration

This repository provides an enhanced anomaly detection and enrichment pipeline for Security Operations Centers (SOC) using LangGraph orchestration, with robust preprocessing, enrichment, and test coverage.

## Directory Structure

```
├── langgraph_ada_integration.py   # Main ADA pipeline (preprocessing, enrichment, detection, etc)
├── test_scale_ai_preprocessing.py # Test script for Scale AI preprocessing (single, batch, large batch)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
```

## Key Features
- **Scale AI Preprocessing:** Deduplication, sensitive field redaction, semantic labeling, kill chain enrichment
- **Batch and Large-Scale Testing:** Easily test with synthetic or real log data
- **Modular & Extensible:** Easily integrate with SIEM/log sources and further enrich

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing Tests
```bash
python test_scale_ai_preprocessing.py
```
- The script will print summary statistics and spot-check results for single, batch, and large-scale log processing.

### 3. Main ADA Pipeline
- Use `langgraph_ada_integration.py` as the core module for your SOC pipeline.
- Integrate with your SIEM/log data as needed.

## Sharing
- All code is self-contained and ready to be pushed to GitLab or shared with your team.
- Remove or update any sensitive credentials/configs before sharing.

## Next Steps
- Integrate with real SIEM/log sources for production-scale validation.
- Extend enrichment logic as needed for your use case.

---

For questions or contributions, please contact the project maintainer.
>>>>>>> master
