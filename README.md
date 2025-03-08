# Loan Interest Rate Prediction

## Problem Understanding
We aim to predict the interest rate category (1/2/3) for a loan application based on client-related information. The goal is to:
- Help consumers understand factors affecting their loan rates to improve creditworthiness.
- Assist lenders with accurate, immediate interest rate category estimations.

## Dataset Overview
The dataset consists of **164,309** entries with **13 features**. Below is a summary of the dataset:

| Feature | Data Type | Null Count | Null Percentage | Unique Values | Sample Unique Values |
|---------|----------|------------|----------------|---------------|----------------------|
| Loan_Amount_Requested | object | 0 | 0.00% | 1290 | [13,250, 23,225] |
| Length_Employed | object | 7371 | 4.49% | 11 | [9 years, 4 years] |
| Home_Owner | object | 25359 | 15.43% | 4 | [Other, nan] |
| Annual_Income | float64 | 25102 | 15.28% | 12305 | [840000.0, 63200.0] |
| Income_Verified | object | 0 | 0.00% | 3 | [VERIFIED - income, not verified] |
| Purpose_Of_Loan | object | 0 | 0.00% | 14 | [moving, medical] |
| Debt_To_Income | float64 | 0 | 0.00% | 3953 | [13.34, 21.77] |
| Inquiries_Last_6Mo | int64 | 0 | 0.00% | 9 | [3, 6] |
| Months_Since_Deliquency | float64 | 88379 | 53.79% | 122 | [129.0, 94.0] |
| Number_Open_Accounts | int64 | 0 | 0.00% | 58 | [8, 1] |
| Total_Accounts | int64 | 0 | 0.00% | 100 | [80, 62] |
| Gender | object | 0 | 0.00% | 2 | [Female, Male] |
| Interest_Rate | int64 | 0 | 0.00% | 3 | [3, 2] |

### Missing Data
- **Months_Since_Delinquency** has a high percentage of missing values (53.79%).
- **Annual_Income** and **Home_Owner** also have notable missing values (~15%).
- Missing values in **Months_Since_Delinquency** may indicate individuals who have never had late payments.

### Data Cleaning
- Removed rows where **Number_Open_Accounts** or **Total_Accounts** is 0, as they likely represent cases with no active or total accounts, which may introduce noise.
- Applied filtering on numerical features to remove extreme outliers:
  - **Annual_Income ≤ 156000.0**
  - **Debt_To_Income ≤ 39.735**
  - **Total_Accounts ≤ 54.5**
  - **Number_Open_Accounts ≤ 23.0**
  - **Inquiries_Last_6Mo ≤ 2.5**

## Class Distribution
| Interest Rate | Count |
|--------------|-------|
| 2 | 48,443 |
| 3 | 40,806 |
| 1 | 23,251 |

## Feature Engineering
New features were created to enhance predictive power:
- **Is_Home_Owner**: 1 if Home_Owner is "Own" or "Mortgage", else 0
- **Is_Income_Verified**: 1 if Income_Verified is "VERIFIED - income", else 0
- **Debt_To_Income_per_Account**: Debt_To_Income divided by Number_Open_Accounts
- **Loan_Request_per_Total_Accounts**: Loan_Amount_Requested divided by Total_Accounts
- **Credit_Utilization**: Number_Open_Accounts divided by Total_Accounts
- **Inquiry_Intensity**: Inquiries_Last_6Mo divided by Number_Open_Accounts
- **Loan_Burden_Ratio**: Loan_Amount_Requested divided by (Annual_Income / 12)
- **Income_Per_Account**: Annual_Income divided by Total_Accounts

After feature engineering, the dataset contained **96,816** entries with **23 features**.

## Exploratory Data Analysis (EDA) Conclusion
- The countplot and density plot show that the data distribution between class 2 and class 3 is quite similar.
- The model may face challenges distinguishing between these two classes effectively.
- Possible solutions: class balancing, advanced feature engineering, or using a model designed for overlapping classes.

## Modeling Preparation
- Converted categorical features into numeric using **One-Hot Encoding** (Gender, Simplified_Purpose) and **Label Encoding** (Interest_Rate).
- Separated the target variable (**Interest_Rate**) from independent variables.

## Model Evaluation
Tested multiple classifiers:
- **Logistic Regression**
- **K-Nearest Neighbors**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **LightGBM** (Best Performing Model)
- **Gaussian Naïve Bayes**
- **Gradient Boosting**
- **AdaBoost**
- **XGBRF**
- **CatBoost**

### Best Model: LightGBM
- **Achieved the highest accuracy** on both training and test data.
- Applying **SMOTE, RFE, and PCA** did not improve model performance.
- **Hyperparameter Tuning Results**:
  - **Best Parameters:**
    ```
    {'subsample': 0.9, 'num_leaves': 50, 'n_estimators': 150, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.9}
    ```
  - **Accuracy of Standard Pipeline:** 52.30%
  - **Accuracy of Best Model from Randomized Search:** 52.42%
  - **The tuned model slightly outperformed the baseline.**

## Deep Learning Experiments
Tested several deep learning models:
- **Feedforward Neural Network (FNN)**
- **Deep Neural Network (DNN)**
- **Convolutional Neural Network (CNN)**

### Results
- None of the deep learning models outperformed LightGBM.
- This aligns with the EDA results, which showed that classes 2 and 3 have very similar characteristics.
- The model's performance remained low due to the difficulty in distinguishing between these two classes.

## Re-Modeling with 2 Classes
- Since classes 2 and 3 were highly similar, they were combined into a single class.
- The same modeling pipeline was applied: model benchmarking, SMOTE, RFE, PCA (though they did not improve performance), and hyperparameter tuning.

### Best Model After Re-Modeling: LightGBM
- **Best Hyperparameters:**
  ```
  {'subsample': 0.9, 'num_leaves': 31, 'n_estimators': 200, 'max_depth': 20, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
  ```
- **Accuracy of Standard Pipeline:** 79.68%
- **Accuracy of Best Model from Randomized Search:** 79.75%
- **The tuned model performed better than the standard pipeline.**



