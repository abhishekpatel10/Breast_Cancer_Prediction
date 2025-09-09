# Breast Cancer Detection using Machine Learning

This project aims to develop and evaluate various machine learning models to accurately classify breast tumors as either benign or malignant. Using the Wisconsin Breast Cancer dataset, this repository provides a comprehensive analysis, from data exploration and preprocessing to model training, evaluation, and interpretation.

## 📋 Table of Contents
* [Project Overview](#-project-overview)
* [About the Dataset](#-about-the-dataset)
* [Technologies & Libraries Used](#-technologies--libraries-used)
* [Installation](#-installation)
* [How to Use](#-how-to-use)
* [Workflow](#-workflow)
* [Models Implemented](#-models-implemented)
* [Evaluation](#-evaluation)
* [Results Summary](#-results-summary)
* [Model Interpretation with SHAP](#-model-interpretation-with-shap)
* [Contributing](#-contributing)
* [License](#-license)

## 📌 Project Overview

The primary goal of this project is to apply supervised machine learning techniques to classify breast cancer diagnoses. The project explores the entire machine learning pipeline:

1.  **Data Exploration & Visualization**: Understanding the features and their distributions.
2.  **Data Preprocessing**: Scaling numerical features to prepare the data for modeling.
3.  **Model Training**: Implementing and training a variety of classification algorithms.
4.  **Hyperparameter Tuning**: Optimizing model performance using `GridSearchCV` and `RandomizedSearchCV`.
5.  **Model Evaluation**: Assessing model performance using a comprehensive set of metrics and visualizations.
6.  **Model Interpretation**: Using SHAP (SHapley Additive exPlanations) to understand the predictions of the best-performing model.

## 💾 About the Dataset

This project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

*   **Source**: The dataset is available in the `scikit-learn` library and originally from the UCI Machine Learning Repository.
*   **Instances**: 569
*   **Features**: 30 numeric, predictive attributes computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image (e.g., radius, texture, perimeter, area).
*   **Target Variable**: `diagnosis` (Malignant or Benign).

## 💻 Technologies & Libraries Used

*   **Python 3.x**
*   **NumPy**: For numerical operations.
*   **Pandas**: For data manipulation and analysis.
*   **Matplotlib** & **Seaborn**: For data visualization.
*   **Scikit-learn**: For data preprocessing, model implementation, and evaluation.
*   **SHAP**: For model interpretability and explaining predictions.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abhishekpatel10/Breast_Cancer_Prediction.git
    cd Breast_Cancer_Prediction
    ```

2.  **Install the required libraries:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file, you can create one or provide the manual installation command:*
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn shap
    ```

## 🚀 How to Use

1.  Ensure all dependencies are installed.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
3.  Open the `.ipynb` file to view and run the code cells sequentially. The notebook is structured to walk through the entire process from data loading to final model interpretation.

## 🌊 Workflow

1.  **Load Data**: The Wisconsin Breast Cancer dataset is loaded.
2.  **Exploratory Data Analysis (EDA)**: The data is explored to understand feature distributions, correlations, and the balance between the target classes.
3.  **Data Preprocessing**:
    *   The data is split into training and testing sets using `train_test_split`.
    *   Features are scaled using `StandardScaler` to normalize the data, which is crucial for distance-based algorithms like SVM and KNN.
4.  **Model Building and Training**: The implemented models are trained on the preprocessed training data.
5.  **Performance Evaluation**: Models are evaluated on the test set using various metrics and visualizations.
6.  **SHAP Analysis**: The best model is analyzed using SHAP to understand which features are most influential in its predictions.

## 🤖 Models Implemented

The following classification models were implemented and evaluated:
*   **Logistic Regression** (with L1, L2, and Elastic Net regularization)
*   **K-Nearest Neighbors (KNN)**
*   **Support Vector Machine (SVM)**
*   **Decision Tree Classifier**
*   **Random Forest Classifier**
*   **Multi-layer Perceptron (MLP) Classifier** (Neural Network)

## 📊 Evaluation

Model performance is assessed using a variety of metrics and visualizations:

*   **Metrics**:
    *   Accuracy
    *   Precision
    *   Recall
    *   F1-Score
    *   ROC AUC Score
*   **Visualizations**:
    *   **Confusion Matrix**: To visualize the number of correct and incorrect predictions for each class.
    *   **ROC Curve**: To illustrate the diagnostic ability of the binary classifier system.
    *   **Precision-Recall Curve**: To evaluate the trade-off between precision and recall for different thresholds.
    *   **Classification Report**: A detailed, per-class breakdown of precision, recall, and F1-score.

## 🏆 Results Summary

After training and hyperparameter tuning, the models were evaluated on the test set. The **Neural Network**, **Support Vector Machine**, and **LASSO (L1) Logistic Regression** models achieved the highest test accuracy of **98.25%**. The Neural Network and SVM also achieved a high F1-score of **95.1%**.

While several models, like Random Forest and K-Nearest Neighbours, achieved 100% training accuracy, this suggests some overfitting. The top-performing models demonstrated a better balance between training and test performance, indicating strong generalization capabilities.

Here is a detailed breakdown of each model's performance:

| Model | Tuning Method | Best Parameters | Train Acc. | Test Acc. | F1-Score (Test) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Neural Network** | Randomized Search | `solver: sgd`, `learning_rate_init: 0.488`, `activation: logistic` | 98.24% | **98.25%** | **95.1%** |
| **Support Vector Machine** | GridSearchCV | `C: 1`, `kernel: rbf`, `gamma: scale` | 98.46% | **98.25%** | **95.1%** |
| **LASSO (Logistic L1)** | Randomized Search | `solver: liblinear`, `C: 0.26` | 98.24% | **98.24%** | **95.1%** |
| **Random Forest** | Randomized Search | `n_estimators: 30`, `max_features: log2`, `max_depth: 8` | 100% | 97.36% | **95.1%** |
| **Ridge (Logistic L2)** | Randomized Search | `solver: liblinear`, `C: 0.22` | 98.46% | 97.36% | 92.7% |
| **Elastic Net (Logistic)** | Randomized Search | `solver: saga`, `l1_ratio: 0.01`, `C: 2.14` | 98.90% | 97.36% | 92.68% |
| **Decision Tree** | Randomized Search | `min_samples_split: 3`, `max_depth: 28`, `criterion: gini` | 100% | 96.5% | 92.7% |
| **K-Nearest Neighbours** | GridSearchCV | `metric: manhattan`, `n_neighbors: 9`, `weights: distance` | 100% | 96.49% | 92.7% |
| **Random Forest (Unscaled)** | Randomized Search | `n_estimators: 120`, `max_features: sqrt`, `max_depth: 7` | 100% | 96.49% | **95.1%** |


## 🧠 Model Interpretation with SHAP

To ensure our model is not just a "black box," we employed the **SHAP (SHapley Additive exPlanations)** library to interpret the output of one of the best-performing models (e.g., SVM or Neural Network).

The SHAP summary plot shows the most important features and their impact on the model's predictions. For example, features like `worst concave points`, `worst perimeter`, and `worst radius` were found to be the most significant predictors for classifying a tumor as malignant.

## 🙌 Contributing

Contributions are welcome! If you have any ideas, suggestions, or find any bugs, please open an issue or submit a pull request.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
