# Multi-Agent SOC Workflow Diagrams

## Complete System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        SIEM[SIEM Events<br/>BigQuery]
        LOGS[Security Logs<br/>Network Traffic]
    end
    
    subgraph "AI Agent Layer"
        ADA[ADA<br/>Anomaly Detection Agent<br/>🔍 Detection & Scoring]
        TAA[TAA<br/>Triage & Analysis Agent<br/>🧠 LLM Analysis]
        CRA[CRA<br/>Containment Response Agent<br/>🛡️ Automated Response]
        CLA[CLA<br/>Continuous Learning Agent<br/>📚 Model Improvement]
    end
    
    subgraph "Orchestration Layer"
        LANG[LangGraph<br/>Workflow Engine<br/>🔄 State Management]
        PUB[Pub/Sub<br/>Message Broker<br/>📡 Async Communication]
    end
    
    subgraph "Data Layer"
        BQ[BigQuery<br/>Data Warehouse<br/>📊 Analytics & Storage]
        FS[Firestore<br/>State Management<br/>💾 Agent State]
        CS[Cloud Storage<br/>Model Artifacts<br/>🗄️ ML Models]
    end
    
    subgraph "Monitoring Layer"
        DASH1[Production CLA Dashboard<br/>📊 Service Monitoring]
        DASH2[TAA-CRA Integration Dashboard<br/>🔗 Integration Status]
        ALERT[Alerting System<br/>🚨 Notifications]
    end
    
    %% Data Flow
    SIEM --> ADA
    LOGS --> ADA
    
    %% Agent Communication
    ADA -->|Alerts| TAA
    TAA -->|Containment Requests| CRA
    TAA -->|Feedback| CLA
    TAA -->|Reports| DASH2
    
    %% Orchestration
    LANG --> ADA
    LANG --> TAA
    LANG --> CRA
    
    %% Message Broker
    PUB --> ADA
    PUB --> TAA
    PUB --> CRA
    PUB --> CLA
    
    %% Data Persistence
    BQ --> ADA
    BQ --> TAA
    BQ --> CLA
    FS --> TAA
    FS --> CRA
    CS --> CLA
    
    %% Monitoring
    ADA --> DASH1
    CLA --> DASH1
    TAA --> DASH2
    CRA --> DASH2
    
    %% Learning Loop
    CLA -->|Model Updates| ADA
    CLA -->|Parameter Updates| TAA
```

## Detailed Agent Interaction Flow

```mermaid
sequenceDiagram
    participant SIEM as SIEM Events
    participant ADA as ADA Agent
    participant TAA as TAA Agent
    participant CRA as CRA Agent
    participant CLA as CLA Agent
    participant BQ as BigQuery
    participant DASH as Dashboard
    
    Note over SIEM, DASH: Real-time Threat Detection & Response
    
    SIEM->>ADA: Raw security events
    ADA->>ADA: Anomaly detection (ML)
    ADA->>BQ: Store processed alerts
    ADA->>TAA: Alert with confidence score
    
    Note over TAA: LangGraph Workflow
    TAA->>TAA: Enrichment & Analysis
    TAA->>TAA: LLM-powered classification
    
    alt High Severity Threat
        TAA->>CRA: Containment request
        CRA->>CRA: Execute containment
        CRA->>BQ: Log containment action
    end
    
    TAA->>CLA: Feedback for learning
    TAA->>BQ: Store analysis results
    TAA->>DASH: Update dashboard
    
    Note over CLA: Continuous Learning
    CLA->>BQ: Poll for new feedback
    CLA->>CLA: Analyze performance
    CLA->>CLA: Retrain models
    CLA->>ADA: Deploy improved model
    CLA->>DASH: Update metrics
```

## TAA LangGraph Workflow Detail

```mermaid
graph TD
    A[Receive Alert] --> B[Enrichment]
    B --> C[LLM Analysis]
    C --> D{Decision Node}
    
    D -->|High Severity + True Positive| E[Containment Node]
    D -->|Low Confidence| F[Manual Review Node]
    D -->|Other Cases| G[Feedback Node]
    
    E --> H[Publish to CRA]
    F --> I[Queue for Human Review]
    G --> J[Publish to CLA]
    
    H --> K[Reporting Node]
    I --> K
    J --> K
    
    K --> L[Publish to RVA]
    L --> M[End]
    
    style E fill:#ff9999
    style F fill:#ffcc99
    style G fill:#99ccff
    style K fill:#99ff99
```

## Pub/Sub Message Flow

```mermaid
graph LR
    subgraph "Publishers"
        ADA_P[ADA Agent]
        TAA_P[TAA Agent]
    end
    
    subgraph "Topics"
        ADA_TOPIC[ada-alerts]
        CONTAIN_TOPIC[containment-requests]
        FEEDBACK_TOPIC[taa-feedback]
        REPORTS_TOPIC[taa-reports]
    end
    
    subgraph "Subscribers"
        TAA_S[TAA Agent]
        CRA_S[CRA Agent]
        CLA_S[CLA Agent]
        RVA_S[RVA Agent]
    end
    
    ADA_P -->|Alerts| ADA_TOPIC
    TAA_P -->|Containment| CONTAIN_TOPIC
    TAA_P -->|Feedback| FEEDBACK_TOPIC
    TAA_P -->|Reports| REPORTS_TOPIC
    
    ADA_TOPIC -->|Subscribe| TAA_S
    CONTAIN_TOPIC -->|Subscribe| CRA_S
    FEEDBACK_TOPIC -->|Subscribe| CLA_S
    REPORTS_TOPIC -->|Subscribe| RVA_S
```

## Data Storage Architecture

```mermaid
graph TB
    subgraph "BigQuery Tables"
        SIEM_TABLE[siem_events<br/>Raw security data]
        ALERTS_TABLE[processed_alerts<br/>ADA detection results]
        FEEDBACK_TABLE[feedback<br/>TAA analysis feedback]
        CONTAIN_TABLE[containment_requests<br/>CRA action requests]
        METRICS_TABLE[model_metrics<br/>CLA performance data]
        INCIDENTS_TABLE[incidents<br/>Complete incident records]
    end
    
    subgraph "Firestore Collections"
        TAA_STATE[taa_state<br/>Agent workflow state]
        CRA_STATE[cra_state<br/>Containment state]
    end
    
    subgraph "Cloud Storage"
        MODELS[model_artifacts<br/>Trained ML models]
        LOGS[service_logs<br/>Application logs]
    end
    
    SIEM_TABLE --> ALERTS_TABLE
    ALERTS_TABLE --> FEEDBACK_TABLE
    FEEDBACK_TABLE --> METRICS_TABLE
    CONTAIN_TABLE --> INCIDENTS_TABLE
```

## Service Deployment Architecture

```mermaid
graph TB
    subgraph "GCP VM: xdgaisocapp01"
        subgraph "Systemd Services"
            ADA_SVC[ada-production.service<br/>Port 8080]
            CLA_SVC[production-cla.service<br/>Port 8080]
            BQ_SVC[ada-bigquery-integration.service]
        end
        
        subgraph "Manual Services"
            CRA_SVC[cra_service_working.py<br/>Pub/Sub Listener]
            TAA_SVC[taa_service.py<br/>Original TAA]
        end
        
        subgraph "Dashboards"
            DASH1[Production CLA Dashboard<br/>Port 8505]
            DASH2[TAA-CRA Integration Dashboard<br/>Port 8531]
        end
    end
    
    subgraph "Google Cloud Platform"
        PUBSUB[Pub/Sub Topics]
        BIGQUERY[BigQuery]
        FIRESTORE[Firestore]
        STORAGE[Cloud Storage]
    end
    
    ADA_SVC --> PUBSUB
    CLA_SVC --> PUBSUB
    CRA_SVC --> PUBSUB
    TAA_SVC --> PUBSUB
    
    ADA_SVC --> BIGQUERY
    CLA_SVC --> BIGQUERY
    CRA_SVC --> FIRESTORE
    
    DASH1 --> ADA_SVC
    DASH1 --> CLA_SVC
    DASH2 --> PUBSUB
```

## Monitoring & Observability Stack

```mermaid
graph TB
    subgraph "Data Collection"
        LOGS[Application Logs]
        METRICS[Performance Metrics]
        EVENTS[System Events]
    end
    
    subgraph "Processing Layer"
        LOG_PROC[Log Processing]
        METRIC_PROC[Metric Aggregation]
        EVENT_PROC[Event Correlation]
    end
    
    subgraph "Storage"
        BQ_LOGS[BigQuery Logs]
        BQ_METRICS[BigQuery Metrics]
        BQ_EVENTS[BigQuery Events]
    end
    
    subgraph "Visualization"
        DASH[Streamlit Dashboards]
        ALERTS[Alert Manager]
        REPORTS[Report Generator]
    end
    
    LOGS --> LOG_PROC
    METRICS --> METRIC_PROC
    EVENTS --> EVENT_PROC
    
    LOG_PROC --> BQ_LOGS
    METRIC_PROC --> BQ_METRICS
    EVENT_PROC --> BQ_EVENTS
    
    BQ_LOGS --> DASH
    BQ_METRICS --> DASH
    BQ_EVENTS --> DASH
    
    DASH --> ALERTS
    DASH --> REPORTS
```

---

## How to Use These Diagrams

1. **Copy the Mermaid code** from any diagram above
2. **Paste into a Mermaid editor** like:
   - [Mermaid Live Editor](https://mermaid.live/)
   - [GitHub/GitLab** (native support)
   - [Notion** (with Mermaid plugin)
   - [VS Code** (with Mermaid extension)

3. **Customize as needed** for your specific documentation or presentations

These diagrams provide a comprehensive visual representation of your multi-agent SOC architecture and can be used in documentation, presentations, or for onboarding new team members.
