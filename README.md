**"Integrated Risk Scoring and Exploit Prediction for Cyber-Physical Power System Vulnerabilities"**  
(*to appers in Springer Energy Informatics Journal, 2025*)

## Overview

Cyber-Physical Power Systems (CPPS) are increasingly exposed to critical software vulnerabilities, but traditional risk metrics (e.g., CVSS) fail to reflect actual exploitation likelihood or operational impact. This work presents:

- A curated dataset of **4,000+ CPPS-related CVEs** (2020–2025), enriched with:
  - CVSSv3 severity scores
  - CISA Known Exploited Vulnerability (KEV) status
  - EPSS probabilistic exploitation scores
  - ICS-CERT advisory metadata
  - Asset criticality and exposure context
- An end-to-end **machine learning pipeline** using XGBoost and Ridge regression to:
  - Predict risk scores and exploit probability
  - Calibrate outputs via real-world exploit signals
  - Apply SHAP explainability for model transparency
- An actionable **risk prioritization framework** for power grid defenders.

## Contents

- `data/`: Preprocessed dataset (`combined_unified.csv`)
- `src/`: Python code for feature extraction, modeling, and evaluation
- `notebooks/`: Exploratory analysis and SHAP visualizations
- `models/`: Trained model weights and calibration artifacts
- `figures/`: Paper-ready charts and priority distribution figures

##  Key Features

- Composite risk model: blends **CVSS, EPSS, KEV**, and CPPS-specific context
- Dual modeling: regression for continuous risk and classification for exploit prediction
- SHAP analysis: interpretable model insights for each CVE
- Open access to dataset and full pipeline to support reproducibility

##  Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{firdous2025cpps,
  author       = {Firdous, Kausar},
  title        = {CASPER\_01\_Code: CPPS Vulnerabilities Dataset and Risk Scoring Pipeline},
  year         = {2025},
  howpublished = {\url{https://github.com/firdous3679/CASPER_01_Code}},
  note         = {Accessed: YYYY-MM-DD}
}
