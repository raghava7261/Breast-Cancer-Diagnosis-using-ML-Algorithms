# Breast-Cancer-Diagnosis-using-ML-Algorithms
Overview
This project implements and compares traditional machine learning algorithms and deep learning models for early detection of breast cancer using the Wisconsin Diagnostic Breast Cancer Dataset. The aim is to evaluate whether deep learning outperforms traditional ML for structured medical data or vice versa.
 
Project Objectives
•	To analyze performance differences between SVM, Random Forest, and Deep Neural Networks (DNN).
•	To identify the most discriminative features for breast cancer classification.
•	To validate findings through statistical significance testing and ROC-AUC analysis.
 
Dataset
•	Source: Wisconsin Diagnostic Breast Cancer Dataset
•	Instances: 569 samples (212 malignant, 357 benign)
•	Features: 30 numerical features derived from FNA images:
o	Measurements: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
o	Each feature includes: Mean, Standard Error, Worst Value
•	Missing Values: None
 
Methodology
1. Preprocessing
•	Feature scaling (Normalization)
•	Outlier detection and removal
•	Multicollinearity checks
2. Models Implemented
•	Support Vector Machine (SVM)
•	Random Forest (RF)
•	Deep Neural Network (DNN) with hyperparameter tuning
3. Evaluation Metrics
•	Accuracy
•	Precision, Recall, F1-score
•	ROC-AUC
•	McNemar’s Test (Statistical significance)
•	Learning curves for bias-variance analysis
 
Experimental Results
Support Vector Machine (SVM)
•	Accuracy: 99.12%
•	AUC: 0.995
•	Exceptional sensitivity and specificity
•	Clinically deployable performance
Random Forest
•	Accuracy: 95.61%
•	AUC: 0.990
•	Good generalization but lower than SVM
Deep Neural Network (DNN)
•	Validation Accuracy: 94.74%
•	Overfitting observed after ~100 epochs
•	Extensive hyperparameter tuning required
 
Key Insights
•	SVM significantly outperformed DNN and RF for tabular medical data (p < 0.05 via McNemar’s test)
•	Concave point features are most predictive of malignancy
•	Deep learning is not always superior, especially for small structured datasets
•	Simpler models are faster, interpretable, and clinically viable

How to Run


Requirements
•	Python 3.8+
•	Libraries:
java
CopyEdit
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow (for DNN)
Steps
1.	Clone the repository or download project files.
2.	Install dependencies:
nginx
CopyEdit
pip install -r requirements.txt
3.	Open the notebook:
go
CopyEdit
jupyter notebook breast_cancer_diagnosis_final-2.ipynb
4.	Run all cells to reproduce results.
 
Results Summary
Model	Accuracy	ROC-AUC
SVM	99.12%	0.995
Random Forest	95.61%	0.990
DNN	94.74%	0.980
 
Conclusion
•	SVM is the best-performing model for this dataset.
•	DNN struggles due to overfitting and dataset size limitations.
•	Clinical recommendations: Use simpler, interpretable models for structured medical data.

