# Real-Time Financial Risk & Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Apache Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=flat&logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![Apache Flink](https://img.shields.io/badge/Apache_Flink-E6526F?style=flat&logo=apacheflink&logoColor=white)](https://flink.apache.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

Fault-tolerant real-time streaming pipeline for financial fraud detection and portfolio risk assessment. Built with Apache Kafka and Flink for sub-second data processing, with AI-powered anomaly detection that reduced fraudulent transaction detection time from hours to seconds.

---

## Key Results

| Metric | Result |
|---|---|
| Fraud detection time | Hours → seconds |
| Portfolio risk assessment accuracy | +23% over batch methods |
| Risk models implemented | VaR, CVaR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown |
| Anomaly detection models | Isolation Forest, LOF (via Scikit-learn + PyOD) |

---

## Architecture

```
Live Market Data Feed
        │
        ▼
┌───────────────┐
│  Apache Kafka │  Ingestion + topic partitioning
│  (Producer)   │  High-throughput message queue
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Apache Flink  │  Stream processing
│ (Processor)   │  Windowed aggregations
│               │  Stateful computations
└───────┬───────┘
        │
   ┌────┴─────┐
   │          │
   ▼          ▼
┌──────┐  ┌──────────────┐
│ AI   │  │  Risk Models │
│Anomaly│  │  VaR / CVaR  │
│Detect│  │  Sharpe Ratio│
└──┬───┘  └──────┬───────┘
   │              │
   └──────┬───────┘
          │
          ▼
    PostgreSQL Storage
          │
          ▼
    Power BI Dashboard
```

---

## Features

**Real-Time Streaming Pipeline**
- Apache Kafka producers ingest live stock market data across multiple topics
- Apache Flink processes high-velocity transaction streams with stateful windowed operations
- Fault-tolerant design with checkpointing and automatic recovery

**AI-Powered Fraud Detection**
- Isolation Forest and Local Outlier Factor models for unsupervised anomaly detection
- PyOD for ensemble outlier scoring across multiple detection strategies
- Model outputs stream directly into alerting and flagging workflows

**Portfolio Risk Models**
- Value at Risk (VaR) — 95th and 99th percentile loss estimation
- Conditional Value at Risk (CVaR) — expected loss beyond VaR threshold
- Sharpe Ratio — risk-adjusted return measurement
- Sortino Ratio — downside deviation-adjusted performance
- Maximum Drawdown — peak-to-trough loss tracking

---

## Tech Stack

| Component | Tool |
|---|---|
| Message Queue | Apache Kafka |
| Stream Processor | Apache Flink |
| Anomaly Detection | Scikit-learn, PyOD |
| Risk Modeling | Python (NumPy, Pandas) |
| Database | PostgreSQL |
| Visualization | Power BI |

---

## Project Structure

```
Real-Time-Fraud-Detection-System/
├── kafka/
│   ├── producer.py           # Market data ingestion
│   ├── consumer.py           # Stream consumer
│   └── config.py
├── flink/
│   ├── stream_processor.py   # Windowed aggregations
│   └── stateful_ops.py
├── models/
│   ├── anomaly_detection.py  # Isolation Forest + LOF
│   ├── risk_models.py        # VaR, CVaR, Sharpe, Sortino
│   └── ensemble.py           # PyOD ensemble scoring
├── storage/
│   ├── schema.sql            # PostgreSQL schema
│   └── loader.py
├── dashboard/
│   └── fraud_risk.pbix       # Power BI report
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/mukuldesai/Real-Time-Fraud-Detection-System
cd Real-Time-Fraud-Detection-System
pip install -r requirements.txt

# Start Kafka
bin/zookeeper-server-start.sh config/zookeeper.properties &
bin/kafka-server-start.sh config/server.properties &

# Run pipeline
python kafka/producer.py &
python flink/stream_processor.py &
python models/anomaly_detection.py
```

---

## Author

**Mukul Desai** — Data Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mukuldesai-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/mukuldesai)
[![Portfolio](https://img.shields.io/badge/Portfolio-mukuldesai.vercel.app-000000?style=flat&logo=vercel&logoColor=white)](https://mukuldesai.vercel.app)
