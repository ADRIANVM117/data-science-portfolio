# NBA Game Outcome Prediction Project Summary

## Project Objective
The objective of this project is to develop a predictive model capable of forecasting the outcome of an NBA game, i.e., whether a team will win (W) or lose (L) based on various team statistics. The main variables used in the model include individual and team statistics such as points scored, field goal percentage, rebounds, and assists.

---

## Data Analysis and Preparation Process

### Understanding the Problem and Context:
The primary goal is to predict the outcome of a game based on statistics like PTS (points scored), FG_PCT (field goal percentage), and PLUS_MINUS (point difference).  
The distribution of the target variable WL (win or loss) is analyzed to check if the data is balanced, and corrective measures (e.g., oversampling techniques) are taken if necessary.

### Variable Evaluation:
- **Direct Variables:** PTS, FG_PCT, and PLUS_MINUS have a direct relationship with the likelihood of winning.  
- **Contextual Variables:** TEAM_ABBREVIATION, MATCHUP, and GAME_DATE.  
- **Redundant Variables:** FGM can be highly correlated with PTS, so its contribution to the model is assessed.

### Distribution of the Target Variable (WL):
The WL variable was found to be balanced with a 50% win and 50% loss ratio.  
The influence of home vs. away games on the outcome is also analyzed.

### Data Visualization and Transformation:
Various graphs were created to analyze the distribution of variables like PTS, REB, AST, and PLUS_MINUS, checking their shape and identifying potential outliers.  
Boxplots and KDE plots were used to detect outliers and evaluate the need for transformations in the variables.

### Relationship Between Variables:
A correlation matrix was computed to identify relationships between predictor variables such as PTS, FGM, and FTM.  
Multicollinearity was checked using the Variance Inflation Factor (VIF) to avoid redundancy between variables.

---

## Model Selection
Based on the previous analysis, models that do not assume a normal distribution were selected, as some variables show skewed distributions. Suitable models for this project include:  
- **Decision Trees and Random Forest:** These models are robust against non-normal distributions and do not require specific distributions for variables.  
- **Logistic Regression:** Suitable for binary classification problems (win or loss).  
- **Neural Networks:** To explore more complex models that can handle non-linear relationships in the data.

---

## Expected Results
The model is expected to predict NBA game outcomes accurately, taking into account key variables that affect team performance. By analyzing variable distributions and selecting appropriate models, the goal is to achieve high performance and accuracy in predictions.

---

## Results Obtained

The results from the model evaluations are as follows:

### 1. **Random Forest Model Evaluation:**
- **Accuracy:** 0.9506  
- **Classification Report:**

  |              | Precision | Recall | F1-score | Support |
  |--------------|-----------|--------|----------|---------|
  | 0            | 0.95      | 0.95   | 0.95     | 10550   |
  | 1            | 0.95      | 0.95   | 0.95     | 10802   |

  - **Accuracy:** 0.95  
  - **Macro avg:** 0.95 (Precision), 0.95 (Recall), 0.95 (F1-score)  
  - **Weighted avg:** 0.95 (Precision), 0.95 (Recall), 0.95 (F1-score)

---

### 2. **Gradient Boosting Model Evaluation:**
- **Accuracy:** 0.9496  
- **Classification Report:**

  |              | Precision | Recall | F1-score | Support |
  |--------------|-----------|--------|----------|---------|
  | 0            | 0.95      | 0.95   | 0.95     | 10550   |
  | 1            | 0.95      | 0.95   | 0.95     | 10802   |

  - **Accuracy:** 0.95  
  - **Macro avg:** 0.95 (Precision), 0.95 (Recall), 0.95 (F1-score)  
  - **Weighted avg:** 0.95 (Precision), 0.95 (Recall), 0.95 (F1-score)

---

### 3. **MLP Model Evaluation:**
- **Accuracy:** 0.8496  
- **Classification Report:**

  |              | Precision | Recall | F1-score | Support |
  |--------------|-----------|--------|----------|---------|
  | 0            | 0.77      | 0.99   | 0.87     | 10550   |
  | 1            | 0.99      | 0.71   | 0.83     | 10802   |

  - **Accuracy:** 0.85  
  - **Macro avg:** 0.88 (Precision), 0.85 (Recall), 0.85 (F1-score)  
  - **Weighted avg:** 0.88 (Precision), 0.85 (Recall), 0.85 (F1-score)

---
![image](https://github.com/user-attachments/assets/e3e73eba-3747-4e37-918c-5f9af1e8d601)

---
This project provides a comprehensive approach to analyzing and predicting NBA game outcomes, using statistical analysis, data cleaning, and advanced machine learning models. The results show high performance from both the Random Forest and Gradient Boosting models, with a slightly lower performance from the MLP model.

