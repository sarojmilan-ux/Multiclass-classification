# 🍄 Mushroom Classification - Machine Learning Project

## Problem Statement

This project aims to classify mushrooms as either **edible** or **poisonous** based on their physical characteristics. Given the potentially fatal consequences of misclassification, accurate prediction is critical. The goal is to implement and compare multiple machine learning classification models to identify the most reliable approach for this binary classification task.

The dataset contains 8,124 mushroom samples with 22 categorical features describing various physical attributes such as cap shape, cap color, odor, gill characteristics, stalk properties, and habitat. The target variable indicates whether a mushroom is edible (e) or poisonous (p).

## Dataset Description

**Source**: [UCI Machine Learning Repository - Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)

### Dataset Statistics:
- **Number of Instances**: 8,124
- **Number of Features**: 22 (all categorical)
- **Target Variable**: Class (edible=e, poisonous=p)
- **Class Distribution**: 
  - Edible: 4,208 (51.8%)
  - Poisonous: 3,916 (48.2%)
- **Missing Values**: Some instances have missing values denoted by '?'

### Features:
1. **cap-shape**: bell, conical, convex, flat, knobbed, sunken
2. **cap-surface**: fibrous, grooves, scaly, smooth
3. **cap-color**: brown, buff, cinnamon, gray, green, pink, purple, red, white, yellow
4. **bruises**: bruises, no
5. **odor**: almond, anise, creosote, fishy, foul, musty, none, pungent, spicy
6. **gill-attachment**: attached, descending, free, notched
7. **gill-spacing**: close, crowded, distant
8. **gill-size**: broad, narrow
9. **gill-color**: black, brown, buff, chocolate, gray, green, orange, pink, purple, red, white, yellow
10. **stalk-shape**: enlarging, tapering
11. **stalk-root**: bulbous, club, cup, equal, rhizomorphs, rooted, missing
12. **stalk-surface-above-ring**: fibrous, scaly, silky, smooth
13. **stalk-surface-below-ring**: fibrous, scaly, silky, smooth
14. **stalk-color-above-ring**: brown, buff, cinnamon, gray, orange, pink, red, white, yellow
15. **stalk-color-below-ring**: brown, buff, cinnamon, gray, orange, pink, red, white, yellow
16. **veil-type**: partial, universal
17. **veil-color**: brown, orange, white, yellow
18. **ring-number**: none, one, two
19. **ring-type**: cobwebby, evanescent, flaring, large, none, pendant, sheath, zone
20. **spore-print-color**: black, brown, buff, chocolate, green, orange, purple, white, yellow
21. **population**: abundant, clustered, numerous, scattered, several, solitary
22. **habitat**: grasses, leaves, meadows, paths, urban, waste, woods

### Data Preprocessing:
- All categorical features were encoded using Label Encoding
- Train-test split: 80% training, 20% testing
- Stratified sampling to maintain class distribution

## Models Used

### Comparison Table - Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| kNN | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

*Note: Run `python model/train_models.py` to generate actual metrics*

### Model Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression provides a baseline linear approach for this binary classification task. It performs well when there are clear linear decision boundaries between classes. Expected to achieve good accuracy on this dataset as mushroom features have strong discriminative power. However, it may not capture complex non-linear relationships between features. |
| Decision Tree | Decision Tree is well-suited for this categorical dataset as it naturally handles categorical features without extensive preprocessing. It creates interpretable rules (e.g., "if odor=foul then poisonous"). May be prone to overfitting without proper depth constraints, but provides excellent insight into which features are most important for classification. |
| kNN | K-Nearest Neighbors is an instance-based learner that classifies based on similarity to training samples. Performance depends heavily on the choice of k and distance metric. May struggle with high-dimensional spaces and categorical features, requiring careful feature encoding. Computationally expensive during prediction phase. |
| Naive Bayes | Multinomial Naive Bayes works well with categorical data and assumes feature independence. Despite the strong independence assumption (which may not hold for mushroom features), it often performs surprisingly well. Fast training and prediction, making it efficient for this dataset. May underperform if features are highly correlated. |
| Random Forest (Ensemble) | Random Forest combines multiple decision trees to reduce overfitting and improve generalization. Expected to achieve one of the highest accuracies on this dataset due to its ensemble nature. Handles categorical features well and provides feature importance rankings. More robust than single decision tree and less prone to overfitting. |
| XGBoost (Ensemble) | XGBoost is a powerful gradient boosting ensemble method known for winning many ML competitions. Expected to achieve the highest or near-highest performance on this dataset. Builds trees sequentially, with each tree correcting errors of previous ones. Handles missing values naturally and provides excellent performance with proper hyperparameter tuning. May require more computational resources than simpler models. |

### General Observations:
- **Ensemble methods** (Random Forest and XGBoost) are expected to outperform single models due to their ability to combine multiple weak learners
- The **mushroom dataset is well-structured** with strong feature-target relationships, so most models should achieve high accuracy (>90%)
- **Odor** is likely to be the most important feature based on domain knowledge (certain odors strongly indicate poisonous mushrooms)
- **Tree-based models** (Decision Tree, Random Forest, XGBoost) should perform best as they naturally handle categorical data
- **Evaluation metrics** like Precision and Recall are crucial here - we want to minimize False Positives (predicting poisonous as edible) as it could be fatal

## Project Structure

```
project-folder/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Model files and training scripts
│   ├── train_models.py            # Training script for all 6 models
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── model_results.csv          # Comparison metrics
│   └── test_data.csv              # Test dataset for deployment
│
└── mushroom/                       # Original dataset
    ├── agaricus-lepiota.data
    └── agaricus-lepiota.names
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the models**
```bash
python model/train_models.py
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Training Models

To train all 6 models and generate evaluation metrics:

```bash
python model/train_models.py
```

This will:
- Load and preprocess the mushroom dataset
- Train all 6 classification models
- Calculate evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Save trained models as `.pkl` files
- Generate `model_results.csv` with comparison metrics
- Create test dataset for the Streamlit app

### Running the Streamlit App

```bash
streamlit run app.py
```

#### App Features:
1. **Model Selection**: Choose from 6 trained models via dropdown
2. **Data Upload**: Upload test data in CSV format
3. **Predictions**: View predictions for all samples
4. **Evaluation Metrics**: See accuracy, AUC, precision, recall, F1, and MCC scores
5. **Confusion Matrix**: Visual representation of classification results
6. **Classification Report**: Detailed per-class performance metrics

## Deployment on Streamlit Community Cloud

### Steps:

1. **Push to GitHub**
   - Ensure all files are committed to your GitHub repository
   - Include `.gitignore` to exclude large files and virtual environments

2. **Deploy on Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch (usually `main`)
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Wait for Deployment**
   - Streamlit will install dependencies from `requirements.txt`
   - App will be live in a few minutes
   - You'll receive a public URL like `https://your-app.streamlit.app`

### Important Notes:
- Ensure `model/` folder with trained models is pushed to GitHub
- Test data should be small (Streamlit free tier has limited storage)
- All dependencies must be listed in `requirements.txt`

## Technologies Used

- **Python 3.8+**
- **scikit-learn**: ML models and metrics
- **XGBoost**: Gradient boosting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

## Evaluation Metrics Explained

1. **Accuracy**: Overall correctness of the model
2. **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
3. **Precision**: Of all positive predictions, how many were correct
4. **Recall**: Of all actual positives, how many were correctly identified
5. **F1 Score**: Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix values

## Future Improvements

- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Feature importance analysis
- [ ] Cross-validation for more robust evaluation
- [ ] Support for real-time image-based mushroom classification
- [ ] Integration with mushroom identification APIs
- [ ] Mobile-responsive UI enhancements

## License

This project is for educational purposes as part of an academic assignment.

## References

- Dataset: [UCI Machine Learning Repository - Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)
- Streamlit Documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
- scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
- XGBoost Documentation: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)

---

**Author**: [Your Name]  
**Course**: Machine Learning  
**Institution**: BITS Pilani  
**Date**: January 2026
