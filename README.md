# Sleep-Score-Prediction
This GitHub repository houses research utilizing Fitbit wearable data on sleep metrics. Datasets include overall sleep scores, composition scores, deep sleep duration, and more. Valuable for analyzing sleep patterns and behaviors.
This Python script involves data preprocessing and regression analysis for predicting sleep scores based on various features. It includes the following steps:

## Data Preprocessing

1. Importing necessary libraries.
2. Loading sleep statistics and sleep score datasets.
3. Preprocessing sleep statistics data by cleaning, converting data types, and handling missing values.
4. Preprocessing sleep score data.
5. Joining both datasets and selecting relevant features.
6. Visualizing relationships between features and sleep scores.
7. Splitting the data into training, testing, and validation sets.
8. Scaling the features.
9. Performing feature selection using Lasso Regression.

## Baseline Estimation

1. Establishing a baseline by predicting sleep scores using median values.

## Regression Analysis

1. Multiple Linear Regression (MLR)
2. Random Forest Regression
3. Extreme Gradient Boosting (XGBoost)

## Cross-Validation

Performing cross-validation to compare model performances.

## Model Evaluation

Evaluating models on the test dataset and comparing their performance metrics.

## Running the Script

1. Ensure you have Python installed on your system along with required libraries.
2. Run the script using Python interpreter.

## Example

The script predicts sleep scores based on various features and evaluates different regression models' performance using cross-validation and test data.

Feel free to modify and integrate this script into your projects as needed.
