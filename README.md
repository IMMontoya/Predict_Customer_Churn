# Customer Churn Prediction

## Description

This project focuses on predicting customer churn for Beta Bank, aiming to help the bank effectively manage and mitigate the loss of clients. The primary goal is to achieve a model with an F1 score of at least 0.59, utilizing additional metrics like AUC-ROC for a comprehensive model evaluation. The project tests a series of models with a limited search space for tuning, along with multiple class balancing methods to determine the best approach for creating a model that identifies potential churners based on historical data reflecting clients' behaviors and interactions with the bank.

[Jupyter Notebook](nb.ipynb)

## Table of Contents

- [Data Description](#data-description)
- [Data Inspection and Preparation](#data-inspection-and-preparation)
- [Initial Model Training Findings](#initial-model-training-findings)
- [Methods for Balancing Target Class](#methods-for-balancing-target-class)
- [Threshold Optimization](#threshold-optimization)
- [Conclusion and Recommendations](#conclusion-and-recommendations)

## Data Description

[The dataset](Churn.csv) provided by Beta Bank includes several features such as the customer's credit score, geography, gender, age, tenure, account balance, number of products, whether they have a credit card, whether they are an active member, estimated salary, and churn flag. This information is used to understand patterns and predict customer behavior towards churning.

- **CreditScore**: The credit score of the customer.
- **Geography**: The country of residence of the customer.
- **Gender**: The gender of the customer.
- **Age**: The age of the customer.
- **Tenure**: The period a customer has been with the bank.
- **Balance**: The account balance.
- **NumOfProducts**: The number of products the customer uses.
- **HasCrCard**: Indicates if the customer has a credit card.
- **IsActiveMember**: Indicates if the customer is an active member.
- **EstimatedSalary**: The estimated salary of the customer.
- **Exited**: Whether the customer has left the bank (target variable).

## Data Inspection and Preparation

- **Missing Values**: About 9% of the data in the Tenure column was NaN. These values were filled with the median values from tenure grouped by the Age column.
- **Dropped Rows**: RowNumber and CustomerID were dropped, since these values are undesirable for training the model.
- **Splitting Data**: the data set was split into 3:1:1 training, validation, and test sets.

## Initial Model Training Findings

Initial training of models was conducted without adjusting for the imbalance in the classes to establish a baseline performance. Three different machine learning algorithms were employed:

- Decision Tree Classifier  
![Decision Tree Classifier ROC Curve](/images/decisiontree_RocCurve.png)  
Achieved an F1 score of 0.539, suggesting it struggles more with precision and recall balance.

- Random Forest Classifier  
![Random Forest Classifier ROC Curve](/images/randomforrest_RocCurve.png)  
Achieved a slightly better F1 score of 0.556, reflecting its ability to better manage false positives and false negatives.

- Logistic Regression  
![Logistic Regression ROC Curve](/images/logisticregression_RocCurve.png)  
Had the lowest F1 score of 0.325, indicating potential issues with either precision or recall, which could be attributed to the class imbalance.

Training without addressing class imbalance affects each model differently. The Random Forest model showed resilience, likely due to its method of averaging decisions across multiple trees which inherently reduces the variance and bias caused by imbalanced classes. In contrast, the Decision Tree and Logistic Regression models exhibited more sensitivity to the imbalance, impacting their F1 scores and ROC curves.

## Methods for Balancing Target Class

- 'class_weight'='balanced' argument in training each model  
- Upsampling the minority class by 3x, which maintained 'Exited'=1 as the minority class, but brought the ratio much closer to 1:1.
- Downsampling the majority class by 0.3X, also maintained 'Exited'=1 as the minority class, but brought the ratio much closer to 1:1.

The best performing target balancing method was determined to by the downsampling method. The best performing model with this method was the RandomForrestClassifier with the parameters: (`n_estimators: 500`, `criterion: gini`, `max_features: sqrt`).  
![ROC curve for best performing model and target balancing method](/images/downsampling_RandomForrestClass_RocCurve.png)

## Threshold Optimization

An ideal threshold for the best performing model and target class balancing method was determined by iterating over the full range of possible thresholds at an interval of .02. The ideal threshold for maximizing the F1 score was determined to be .54.

## Conclusion and Recommendations

After exploring various methods to balance the classes, the most effective technique was found to be downsampling the majority class by 0.3x. Among the tested models, the Random Forest Classifier, configured with specific parameters: (`n_estimators: 500`, `criterion: gini`, `max_features: sqrt`), outperformed others, achieving an optimal balance with a decision threshold of 0.54. This model not only surpassed the F1 score target but also showed robust performance in terms of AUC-ROC and overall accuracy:

- **F1 Score**: 0.62
- **AUC-ROC Score**: 0.77
- **Accuracy Score**: 0.82

These results indicate that the Random Forest model, combined with the downsampling of the majority class, provides a reliable and effective approach to predicting customer churn at Beta Bank.
