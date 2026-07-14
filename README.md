# 5G Security and Resilience Analysis Framework using Open5GS & UERANSIM

A comprehensive framework for analyzing the security and resilience of a virtualized **5G Core Network** using **Open5GS** and **UERANSIM**. The project evaluates the impact of multiple network attacks on both the **control plane** and **user plane**, collects telemetry data, applies machine learning for attack detection, and generates detailed analytical reports.

---

## Project Overview

Modern 5G networks rely on cloud-native architectures and virtualized network functions, making them vulnerable to sophisticated cyber attacks.

This project simulates different attack scenarios on an Open5GS testbed and analyzes their impact using network telemetry including:

* Round Trip Time (RTT)
* Packet Jitter
* Packet Loss
* CPU Utilization
* Memory Utilization
* Registration Delay

The collected telemetry is used for:

* Statistical analysis
* Visualization
* Rule-based attack detection
* Machine Learning based attack classification
* Anomaly detection for unknown attacks

---

## Features

* Open5GS + UERANSIM based virtual 5G network
* Control Plane and User Plane telemetry analysis
* Automatic parsing of:

  * Ping logs
  * AMF registration logs
  * Resource monitoring logs
* Multiple visualization dashboards
* Random Forest attack classification
* Isolation Forest anomaly detection
* Local Outlier Factor (LOF)
* Automated PDF report generation
* CSV export of collected metrics

---

## Network Architecture

```
               +----------------------+
               |      UERANSIM        |
               |   UE + gNodeB        |
               +----------+-----------+
                          |
                   N2 / N3 Interfaces
                          |
        ---------------------------------------
        |             Open5GS Core            |
        |-------------------------------------|
        | AMF | SMF | UPF | NRF | AUSF | UDM |
        ---------------------------------------
                          |
                 Telemetry Collection
                          |
          -------------------------------
          | RTT | CPU | Memory | AMF Logs |
          -------------------------------
                          |
                 Analysis Framework
                          |
       Visualization + ML Detection + Reports
```

---

## Attack Scenarios

### 1. Normal Traffic

Baseline network performance used for comparison.

Collected metrics:

* RTT
* Packet Loss
* CPU Usage
* Registration Delay
* Jitter

---

### 2. ICMP Flood Attack

A high-rate ICMP Echo Request flood targeting the UE.

Observed effects:

* Increased RTT
* High jitter
* Packet loss
* CPU utilization increase

---

### 3. UDP Flood Attack

A UDP flood targeting the GTP-U tunnel (Port 2152).

Observed effects:

* User plane congestion
* High packet loss
* CPU spikes
* Increased latency

---

### 4. Registration Flood Attack

Repeated fake UE registration requests targeting the AMF.

Observed effects:

* Control plane congestion
* Increased registration delay
* AMF CPU exhaustion

---

### 5. Unknown / Stealthy Attack

Low-rate attack that avoids traditional threshold detection.

Detected using:

* Isolation Forest
* Local Outlier Factor

---

## Telemetry Parameters

The framework extracts:

| Parameter          | Description                                               |
| ------------------ | --------------------------------------------------------- |
| RTT                | Round Trip Time                                           |
| Packet Loss        | Lost packets (%)                                          |
| Jitter             | Variation between RTT samples                             |
| CPU Usage          | Host CPU utilization                                      |
| Memory Usage       | Host memory utilization                                   |
| Registration Delay | Time between Registration Request and Registration Accept |

---

## Machine Learning Pipeline

### Rule-Based Detection

The framework first evaluates:

* RTT thresholds
* Packet Loss thresholds
* CPU thresholds
* Registration delay thresholds

---

### Random Forest Classifier

Classifies traffic into:

* Normal
* ICMP Flood
* UDP Flood
* Registration Flood

Input features:

* RTT
* Jitter
* Packet Loss
* CPU Usage

---

### Anomaly Detection

Unknown attacks are detected using:

* Isolation Forest
* Local Outlier Factor (LOF)

This enables identification of previously unseen attack patterns.

---

## Generated Visualizations

The framework automatically generates:

* RTT Time Series
* Latency CDF
* Box Plot
* Heatmap
* Packet Loss Comparison
* Jitter Timeline
* CPU vs RTT Correlation
* Registration Delay Comparison
* Attack Impact Dashboard
* Control Plane Gantt Chart

---

## Project Structure

```
5G-Security-Open5GS-UERANSIM/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── latency_analyzer.py
│   └── edgecase_refinement.py
│
├── outputs/
│   ├── baseline/
│   │   ├── latency_line_plot.png
│   │   ├── latency_cdf_plot.png
│   │   ├── latency_box_plot.png
│   │   └── latency_heatmap_plot.png
│   │
│   ├── attack_analysis/
│   │   ├── attack_impact_dashboard.png
│   │   ├── packet_loss_comparison.png
│   │   ├── cpu_vs_latency_scatter.png
│   │   ├── jitter_vs_time.png
│   │   ├── before_vs_during_attack.png
│   │   ├── registration_delay_comparison.png
│   │   └── control_plane_gantt.png
│   │
│   └── edgecase_validation/
│       ├── realistic_correlation_matrix.png
│       ├── cpu_vs_registration_delay_realistic.png
│       ├── improved_unknown_attack.png
│       ├── improved_anomaly_score.png
│       └── edgecase_realism_validation_dashboard.png
│
├── reports/
│   ├── edgecase_summary.csv
│   ├── security_analysis_summary.csv
│   ├── security_analysis_report.pdf
│   └── project_report.pdf
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/thanmayee-shetty/5G-Security-Open5GS-UERANSIM.git

cd 5G-Security-Open5GS-UERANSIM
```

Install dependencies

```bash
pip install matplotlib numpy psutil scikit-learn reportlab
```

---

## Running

Example

```bash
python latency_analyzer.py
```

The framework automatically:

* Parses telemetry logs
* Computes metrics
* Detects attacks
* Trains ML model
* Generates plots
* Creates CSV summaries
* Generates a PDF report

---

## Technologies Used

* Python
* Open5GS
* UERANSIM
* NumPy
* Matplotlib
* Scikit-learn
* ReportLab
* Regular Expressions
* psutil

---

## Results

The framework successfully distinguishes between:

* Normal Traffic
* ICMP Flood
* UDP Flood
* Registration Flood
* Unknown anomalies

using a combination of:

* Rule-based detection
* Supervised learning
* Unsupervised anomaly detection

while providing detailed telemetry visualization and automated reporting.

---

## Future Enhancements

* Real-time telemetry streaming
* Prometheus integration
* Grafana dashboards
* Deep Learning based anomaly detection
* Kubernetes deployment
* Multi-gNodeB experiments
* Support for additional 5G attack scenarios

---

## Authors

* **Thanmayee Shetty**
* **Ramya Rao**
* **Anagha Nadgouda**

Guided by **Vidya H**

Department of Computer Science & Engineering

KLE Technological University, Hubballi

---

## License

This project is intended for educational and research purposes.
