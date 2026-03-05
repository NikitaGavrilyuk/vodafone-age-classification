# Telecom Subscriber Age Prediction (Multiclass Classification)

**Language:** Python  
**Infrastructure:** AWS (EC2 / SageMaker / S3)  
**Domain:** Telecommunications, Machine Learning, Ad-Tech Risk Control  

---

## 1. Project Overview

This project implements an end-to-end machine learning pipeline for predicting subscriber age groups based on telecommunication behavioral data.

The task is formulated as a **6-class multiclass classification problem**, where each subscriber is assigned to one of six predefined age categories using:

- Call frequency patterns  
- Call duration statistics  
- Internet traffic usage  
- Temporal behavioral signals (weekdays/weekends, in/out traffic, etc.)

The solution was designed for a large telecommunications company to support marketing optimization and risk-controlled ad targeting.

---

## 2. Business Problem

Understanding subscriber demographics enables:

- Personalized tariff optimization  
- Targeted marketing campaigns  
- Customer segmentation refinement  
- Safer monetization of traffic  

### Critical Business Risk

Incorrect age classification may lead to high-risk ad placements (e.g., 18+ ads shown to minors).

Therefore, beyond accuracy, the model was designed with a **Brand Safety constraint**, prioritizing elimination of critical misclassifications.

---

## 3. Dataset Description

- ~210,000 subscribers  
- 1200+ engineered behavioral features  
- Aggregated telecom usage metrics  
- Highly correlated traffic features  
- Strong overlap between adjacent age groups  

The dataset required dimensionality reduction and feature selection to prevent overfitting and ensure model stability.

---

## 4. Machine Learning Approach

### Problem Formulation

- Multiclass classification (6 age groups)  
- One-vs-Rest evaluation for ROC-AUC  
- Macro-averaged metrics to avoid majority-class bias  

---

### Data Processing Pipeline

1. Data cleaning and validation  
2. Outlier handling  
3. Correlation-based feature filtering  
4. Feature importance–based dimensionality reduction  
5. Train/validation split with stratification  

---

### Handling High Dimensionality

- Removed highly correlated features using correlation thresholding  
- Applied feature importance filtering (tree-based models)  
- Reduced feature space while preserving predictive signal  

---

### Model Selection

The following models were evaluated:

- HistGradientBoosting  
- LightGBM  
- XGBoost  

After systematic comparison and hyperparameter tuning, **XGBoost** demonstrated the best balance between generalization and stability on validation data.

---

## 5. Model Performance

- **Macro-averaged ROC-AUC:** ~0.80  
- **Accuracy:** ~46%  
- Random baseline (6 classes): ~16–17%  

This represents nearly a **3× improvement over random guessing**.

---

### Confusion Matrix Insights

The primary misclassifications occurred between **adjacent age groups** (e.g., 25–34 vs 35–44), indicating behavioral similarity in telecom usage patterns.

Non-adjacent class confusion was significantly lower, confirming the model captured meaningful demographic structure.

![Confusion Matrix](data/Confusion%20matrix.jpg)

---

## 6. Brand Safety Layer (Business Risk Mitigation)

To eliminate critical errors (e.g., minors classified as 18+), a post-processing probability thresholding strategy was implemented.

This approach:

- Adjusted decision thresholds for high-risk classes  
- Prioritized safety over raw accuracy  
- Fully eliminated critical placement errors  

### Result:

- Critical misclassification rate reduced from **26.03% → 0.00%**

This demonstrates alignment between machine learning optimization and real-world business constraints.

 ![Threshold Selection](data/Treshhold%20Selection.jpg)

---

## 7. AWS Infrastructure

The entire pipeline was developed and executed in a cloud environment:

- EC2 – model training  
- SageMaker – experimentation  
- S3 – dataset storage  
- Boto3 – AWS integration  

The architecture ensures scalability and reproducibility.

---

## 8. Repository Structure

```
.
├── data/                   
│   ├── cleaned_dataset.csv
│   ├── cleaned_dataset_stratified.csv
│   ├── Confusion matrix.jpg
│   ├── Treshhold Selection.jpg
│   └── final_scaled_dataset.csv
├── notebooks/             
│   ├── 01_data_cleaning_and_basic_eda.ipynb
│   ├── 02_feature_engineering_and_baselines.ipynb
│   ├── 03_model_tuning_and_error_analysis.ipynb
│   └── Final_Report.ipynb
├── .gitignore              
├── README.md              
└── requirements.txt  
```

### Notebook Breakdown

**01_data_cleaning_and_basic_eda.ipynb**
- Data profiling  
- Distribution analysis  
- Correlation inspection  
- Initial insights  

**02_feature_engineering_and_baselines.ipynb**
- Feature selection  
- Baseline model evaluation  
- Comparative performance analysis  

**03_model_tuning_and_error_analysis.ipynb**
- Hyperparameter tuning (RandomizedSearchCV)  
- Threshold optimization  
- Business risk analysis  

**Final_Report.ipynb**
- Executive-level summary  
- Strategic conclusions  

---

## 9. How to Run

The project was developed in AWS and is best reproduced in a similar cloud environment.

### Steps:

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

Ensure AWS credentials are configured if loading data from S3.

Execute notebooks sequentially:

1. 01 → 02 → 03  
2. Review Final_Report.ipynb  

---

## 10. Key Challenges

- High dimensionality (1200+ features)  
- Multicollinearity among telecom metrics  
- Behavioral similarity between adjacent age groups  
- Severe business penalties for specific misclassifications  
- Trade-off between model accuracy and safety constraints  

---

## 11. Future Improvements

### Advanced Feature Engineering
- Behavioral interaction features  
- Traffic ratios across app categories  
- Temporal segmentation (day/night activity)  
- Deep aggregation features  

### Model Ensembling
- Stacking XGBoost, LightGBM, HistGradientBoosting  
- Meta-model calibration  

### Probability Calibration
- Platt scaling or isotonic regression  
- Improved threshold tuning for safer monetization  

---

## 12. Author

**Nikita Havryliuk**

- LinkedIn: https://www.linkedin.com/in/nikita-havryliuk-7a813b3b5  
- GitHub: https://github.com/NikitaGavrilyuk  
- Email: gavrilyuk.nikita4@gmail.com  

Feel free to reach out regarding technical details, AWS architecture, or business strategy behind this solution.