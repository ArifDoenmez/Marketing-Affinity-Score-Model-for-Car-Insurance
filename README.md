This project focuses on analyzing customer data to build a predictive model that identifies customers with a high affinity for purchasing a car insurance policy. The goal is to leverage data science and machine learning (ML) techniques in R to help focus marketing efforts on the most promising leads.

The entire analysis, from data exploration to final model evaluation, is contained in the R script: `marketing_affinity_score_model.R`.

---

## Table of Contents
* [Project Overview](#project-overview)
* [Dataset Description](#dataset-description)
* [Technical Implementation](#technical-implementation)
  * [1. Data Preparation and EDA](#1-data-preparation-and-eda)
  * [2. Feature Engineering](#2-feature-engineering)
  * [3. Model Building & Training](#3-model-building--training)
  * [4. Model Evaluation](#4-model-evaluation)
* [Potential Improvements](#potential-improvements)
* [How to Run This Project](#how-to-run-this-project)

---

## Project Overview

The objective of this project is to develop a robust classification model that predicts a customer's interest in a car insurance offer. By accurately scoring customer affinity, marketing campaigns can be made more efficient. This solution uses the R ecosystem, leveraging packages like `data.table` for performance and `caret` for a structured modeling workflow.

## Dataset Description

The analysis is based on a dataset provided in three separate `.csv` files, which are merged to create a comprehensive customer profile.

#### `alter_geschlecht.csv`
-   **ID**: Unique Customer ID
-   **Geschlecht**: Customer's Gender
-   **GebDat**: Customer's Date of Birth

#### `rest.csv`
-   **ID**: Unique Customer ID
-   **Fahrerlaubnis**: Indicates if the customer has a driver's license (1 = Yes)
-   **Regional_Code**: A unique code for the customer's residential region
-   **Vorversicherung**: Indicates if the customer already has a car insurance policy (1 = Yes)
-   **Alter_Fzg**: Age of the vehicle in years
-   **Vorschaden**: Indicates if the customer has had a previous vehicle claim (1 = Yes)
-   **Jahresbeitrag**: The expected annual premium in €
-   **Vertriebskanal**: A code identifying the sales/distribution channel
-   **Kundentreue**: Customer loyalty, measured in days since the relationship began

#### `interesse.csv`
-   **ID**: Unique Customer ID
-   **Interesse**: Indicates if the customer is interested in an offer (1 = Yes) **[Target Variable]**

---

## Technical Implementation

The project follows a structured data science and ML methodology implemented entirely in R.

### 1. Data Preparation and EDA
- **Data Integration:** The three source files were efficiently loaded and merged into a single `data.table` object.
- **Data Cleaning:** The `Jahresbeitrag` column was converted to a numeric type.
- **Exploratory Data Analysis (EDA):** A key finding from the EDA was the **imbalance in the target variable (`Interesse`)**, with significantly more "No" instances than "Yes". This observation guided the choice of stratified data splitting and the use of AUC as a primary evaluation metric.

### 2. Feature Engineering
To prepare the data for modeling, several features were transformed and created:
- **Customer Age:** A numerical `Age` feature was engineered from the `GebDat` (Date of Birth) column.
- **Missing Value Imputation:** Missing values in the `Age` and `Jahresbeitrag` columns were imputed using their respective **median** values.
- **Categorical & Ordinal Encoding:**
  - Binary features (`Vorschaden`, `Fahrerlaubnis`, `Vorversicherung`) were converted to factors with levels 0 and 1.
  - The `Alter_Fzg` (Vehicle Age) feature was encoded as an **ordered factor** (`< 1 Year` < `1-2 Year` < `> 2 Years`) to preserve its natural sequence.
  - All other character columns were converted to factors.

### 3. Model Building & Training
- **Model Choice:** A **Random Forest** was selected as the classification algorithm, implemented via the highly efficient `ranger` package.
- **Training Framework:** The `caret` package was used to manage the entire modeling workflow, including data splitting, cross-validation, and model tuning.
- **Data Splitting:** The data was split into an 80% training set and a 20% testing set. **Stratified sampling** (`createDataPartition`) was used to ensure the class distribution was preserved in both sets, which is crucial for the imbalanced target.
- **Training Process:** The model was trained using **5-fold cross-validation**. The training process was configured to optimize for the **Area Under the ROC Curve (AUC)**, a robust metric for imbalanced classification tasks. `caret`'s `preProcess` function was also used to center and scale numerical predictors during training.

### 4. Model Evaluation
The final trained model was rigorously evaluated on the held-out test set.
- **Performance Metrics:** The model's quality was assessed by calculating:

  - The **AUC score** on the test data using the `pROC` package.
  - A detailed **Confusion Matrix** (`caret::confusionMatrix`), providing statistics like Accuracy, Precision, Recall, and F1-score for the positive class ("Yes").
- **Visualizations:** The **ROC curve** was plotted to visualize the trade-off between the true positive rate and false positive rate.

---

## Potential Improvements

- **Hyperparameter Tuning:** Systematically tune the `ranger` model's hyperparameters (e.g., `mtry`, `min.node.size`, `splitrule`) using `caret`'s `tuneGrid` to potentially improve performance.
- **Handling Class Imbalance:** Experiment with advanced techniques like SMOTE (`smotefamily` package) or adjusting class weights within the `train` function to give more importance to the minority class.
- **Alternative Models:** Evaluate other powerful models like XGBoost (`xgboost`) or LightGBM (`lightgbm`) within the `caret` framework to compare performance.

---

## How to Run This Project

### Prerequisites
- R (version 4.0 or higher)
- RStudio (Recommended)

### Installation
1.  Clone this repository to your local machine:
    ```bash
    git clone github.com/ArifDoenmez/Marketing-Affinity-Score-Model-for-Car-Insurance
    cd Marketing-Affinity-Score-Model-for-Car-Insurance
    ```
2.  Open the project in RStudio.
3.  Install the required R packages by running the following command in the R console:
    ```R
    install.packages(c("data.table", "cli", "ggplot2", "lubridate", "caret", "ranger", "pROC", "e1071"))
    ```

### Execution
1.  Place the three data files (`alter_geschlecht.csv`, `rest.csv`, `interesse.csv`) in the project's root directory or a `data/` subfolder (and adjust the file paths in the script if necessary).
2.  Open the `marketing_affinity_score_model.R` file in RStudio.
3.  Run the script from top to bottom to execute the complete analysis pipeline, from data loading to final model evaluation and feature importance plotting.
