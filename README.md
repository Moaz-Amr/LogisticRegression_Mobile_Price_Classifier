# Mobile Price Range Classification using Logistic Regression

## Short Description

This project classifies mobile phones into different price ranges based on their features using the `mobile_price_train.csv` dataset. It employs a Logistic Regression model after performing feature selection based on correlation and handling outliers using the IQR method with median replacement.

## Project Overview

The notebook `day5/Untitled22.ipynb` (consider renaming it to something more descriptive like `mobile_price_classification.ipynb`) performs the following steps:

1.  **Load Data:** Loads the training (`mobile_price_train.csv`) and testing (`mobile_price_test.csv`) datasets. *Note: The provided test dataset (`df_test`) is loaded but not used in the subsequent model training or evaluation.*
2.  **Exploratory Data Analysis (EDA):** Visualizes feature correlations using a heatmap.
3.  **Feature Selection:** Drops several features based on correlation analysis or other criteria.
4.  **Outlier Handling:**
    *   Detects outliers in the remaining numerical features using the Interquartile Range (IQR) method.
    *   Replaces detected outliers with the median value of the respective feature.
5.  **Model Training & Evaluation:**
    *   Splits the **training data** into training and validation sets (`train_test_split`).
    *   Trains a `LogisticRegression` model from `sklearn.linear_model` on the processed training data.
    *   Predicts price ranges on the validation set.
    *   Evaluates the model using the accuracy score.
    *   Generates and plots a confusion matrix.
    *   Visualizes the correlation of the final features with the target variable (`price_range`).

## Dataset

*   **mobile_price_train.csv:** Training data containing various features of mobile phones and their price range category.
*   **mobile_price_test.csv:** Testing data (loaded but not used for evaluation in this notebook).

## Model Used

*   **Logistic Regression:** A linear model commonly used for classification tasks.

## Requirements

*   Python 3
*   NumPy
*   Pandas
*   Scikit-learn
*   Matplotlib
*   Seaborn
*   TensorFlow/Keras (imported but only `mnist` dataset functionality seems intended, which isn't used here - can likely be removed)

## How to Run

1.  Ensure you have the required libraries installed (`pip install numpy pandas scikit-learn matplotlib seaborn`).
2.  Make sure the `mobile_price_train.csv` and `mobile_price_test.csv` files are in the correct path relative to the notebook.
3.  Run the Jupyter Notebook `day5/Untitled22.ipynb`.
