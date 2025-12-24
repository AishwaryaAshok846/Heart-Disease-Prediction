# Heart Disease Prediction Using Machine Learning

Machine learning classification project predicting heart disease presence using patient clinical data with multiple algorithms.

## Project Overview

This project develops and compares machine learning models to predict heart disease based on patient health indicators. Using a dataset of 918 patients, the analysis identifies key risk factors and builds classification models to assist in early disease detection.

## Dataset

- **Source**: [Heart Failure Prediction Dataset by fedesoriano (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Records**: 918 patients
- **Features**: 11 clinical parameters (age, sex, chest pain type, blood pressure, cholesterol, exercise test results, etc.)
- **Target**: Binary classification (disease presence: 0=No, 1=Yes)

## Key Questions

1. What are the strongest predictors of heart disease?
2. How do different patient characteristics relate to disease presence?
3. Which machine learning model performs best for heart disease prediction?
4. How well can we identify at-risk patients using clinical measurements?

## Models Implemented

- **Logistic Regression** (baseline): 85.3% accuracy
- **Random Forest Classifier**: 88.6% accuracy (Best performer)
- **XGBoost Classifier**: 84.8% accuracy

## Key Results

- **Best Model**: Random Forest with 88.6% accuracy and 0.94 AUC score
- **Top Predictive Features**: ST_Slope (24%), Oldpeak (13%), MaxHR (11%)
- **Recall**: 89% - successfully identifies most heart disease cases
- **Precision**: 91% - high confidence when predicting disease

## Key Insights

- Exercise stress test measurements are the strongest predictors of heart disease
- ST_Slope, Oldpeak, and maximum heart rate achieved during exercise are most important
- Exercise-related features outperform traditional risk factors like cholesterol and blood pressure
- Random Forest captures complex patterns better than linear models for this dataset
- The model achieves strong performance while maintaining balance between catching disease cases and minimizing false alarms

## Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

## Project Structure
```
├── heart_disease_prediction.ipynb    # Main analysis notebook
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
```

## How to Run

1. Clone this repository
```bash
git clone https://github.com/AishwaryaAshok846/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open the notebook
   - Locally: `jupyter notebook heart_disease_prediction.ipynb`
   - Or upload to Google Colab

4. Run all cells sequentially

## Methodology

1. **Exploratory Data Analysis**
   - Target variable distribution
   - Correlation analysis with heatmaps
   - Feature distributions by disease status
   - Categorical feature analysis

2. **Data Preprocessing**
   - Handled missing cholesterol values (replaced zeros with median)
   - Encoded categorical variables (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
   - Train-test split (80-20)
   - Feature scaling using StandardScaler

3. **Model Training & Evaluation**
   - Trained three classification models
   - Evaluated using accuracy, precision, recall, F1-score
   - Compared models using confusion matrices and ROC curves
   - Analyzed feature importance

## Visualizations

The notebook includes:
- Correlation heatmaps
- Box plots for continuous features
- Categorical feature distributions
- Model comparison bar charts
- Confusion matrices
- ROC curves with AUC scores
- Feature importance rankings

## Limitations

- 172 zero cholesterol values (18.7%) were replaced with median, potentially introducing bias
- Relatively small dataset (918 patients) may limit generalization
- Does not account for medication use, family history, or lifestyle factors
- Model requires validation on external datasets before clinical use
- Intended as a decision support tool, not a replacement for professional medical diagnosis

## Healthcare Implications

This model demonstrates potential for early heart disease screening in clinical settings. With 89% recall and 91% precision, it can effectively identify high-risk patients while minimizing false alarms. The emphasis on exercise test results highlights the value of stress testing in cardiovascular assessment.

## Future Work

- Validate model on larger, more diverse patient populations
- Incorporate additional features like family history and lifestyle factors
- Develop interactive dashboard for clinical use
- Explore deep learning approaches with larger datasets
- Adjust for demographic and geographic variations

## Author

**Aishwarya Ashok**  
3rd Year Computer Science Student

## Acknowledgments

Dataset from Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano

---

**Note**: This is an educational project demonstrating machine learning applications in healthcare. Any real-world medical application would require validation, regulatory approval, and integration with existing clinical workflows.
