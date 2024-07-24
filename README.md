# Healthcare Stroke Prediction

This project aims to build and evaluate machine learning models to predict the likelihood of a stroke based on various health-related features. The project includes data preprocessing, exploratory data analysis, model building, and hyperparameter tuning.

## Content

1. **Load the Data**
   - Import libraries
   - Load the datasets
  
2. **Overview of the Data**
   - Descriptive Statistics
   - Missing Values
  
3. **Exploratory Data Analysis**
   - Create list of columns by data type
   - Check the distribution of the target class
   - Check the distribution of every feature
   - Check how different numerical features are related to the target class
  
4. **Data Preparation**
   - Data Cleaning
   - Feature Encoding
   - Split X & y
   - Feature Scaling
   - Train-Test split
  
5. **Model Building**
   - Train Model
   - Model Prediction
   - Model Evaluation
  
6. **Improve Model**
   - Handle Class Imbalance
   - Hyperparameter Tuning
   - Save the Final Model

## Setup

To run this project, ensure you have the following libraries installed:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- lightgbm
- imblearn
- pickle

## Instructions

1. **Load the Data**
   - Load the dataset from the provided CSV file.

2. **Overview of the Data**
   - Perform descriptive statistics and identify missing values.

3. **Exploratory Data Analysis**
   - Analyze the distribution of features and the target variable.

4. **Data Preparation**
   - Clean the data, handle missing values, encode categorical features, and scale numerical features.

5. **Model Building**
   - Train and evaluate machine learning models such as Decision Tree, Random Forest, XGBoost, and LightGBM.

6. **Improve Model**
   - Handle class imbalance using oversampling and tune hyperparameters using GridSearchCV.

## Usage

1. Modify the input parameters in the script as needed (file paths, target class, encoding techniques, etc.).
2. Run the script to load the data, preprocess it, build models, and evaluate them.
3. Save the best model for future use.

## Example

To train and evaluate a RandomForestClassifier, set `input_ml_algo` to 'RandomForestClassifier' and run the script. The script will output evaluation metrics and save the trained model to disk.

## File Structure

- **AvinVM_Datascience_final_proj.ipynb**: Jupyter notebook containing the entire workflow.
- **README.md**: This README file.
- **healthcare-dataset-stroke-data.csv**: Input dataset file (replace with your actual file path).
- **final_model.sav**: Saved model file after training and hyperparameter tuning.
