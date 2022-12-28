# Model Card

The model card contains a description of the model that was used to predict income from census data

## Model Details

 - Developer: Orestas Dulinskas
 - Model date: 28/12/2022
 - Model Version: 1.0.0
 - Model type: RandomForestClassifier from scikit-learn

## Intended Use

 - Primary intended use: Predict if income is above or under 50K. 
 - Primary intended users: Economics or sociology researchers. 
 - Use cases outside the scope: Predictions of actual income or thresholds that are not $50K.

## Training Data

The datasets used in the project were the Census Income Data Set from UCI Machine Learning Repository. A binary classification model was designed to determine whether or not a person's income is over 50K based on several features. 
Preprocessing (EDA.ipynb)
 - 80% of total samples
 - Drop NaN
 - Remove all extra space in string
 - All categorical columns are encoded using OneHotEncoder from scikit-learn
 - Label column ('salary') is encoded with LabelBinarizer from scikit-learn

## Evaluation Data

The datasets used in the project were the Census Income Data Set from UCI Machine Learning Repository. A binary classification model was designed to determine whether or not a person's income is over 50K based on several features.
Preprocessing: (can be checked in EDA.ipynb)
- 20% of total samples
- Drop NaN
- Remove all extra space in string   - All categorical column are encoded using one hot encoding
- All categorical column are encoded using OneHotEncoder from scikit-learn
- Label column ('salary') is encoded using LabelBinarizer from scikit-learn

## Metrics

Model performance measures:
  - precision: 0.737
  - recall: 0.626
  - fbeta: 0.677

## Ethical Considerations

There is no sensitive information. There is no use of the data to inform decisions about matters important to human well-being - such as health or safety.

## Caveats and Recommendations

Data slicing analysis (can be checked in /screenshots/slice_output.txt), suggest that model performance might be poor for:
 - Native-country: Puerto-Rico, Cuba, China, Vietnam, Yugoslavia, Dominican-Republic, Honduras, Hongkong, Iran
 - Marital-status: Separated
 - Education: 5th-6th
