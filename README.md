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

| ML Model Name            | Accuracy    | AUC         | Precision   | Recall      | F1          | MCC         |
| ------------------------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Logistic Regression      | 0.955076923 | 0.982118534 | 0.955138405 | 0.955076923 | 0.955063799 | 0.910074658 |
| Decision Tree            | 1           | 1           | 1           | 1           | 1           | 1           |
| kNN                      | 0.996923077 | 0.999987866 | 0.996930272 | 0.996923077 | 0.996923273 | 0.993845649 |
| Naive Bayes              | 0.812923077 | 0.897307087 | 0.829276874 | 0.812923077 | 0.809555481 | 0.639886374 |
| Random Forest (Ensemble) | 1           | 1           | 1           | 1           | 1           | 1           |
| XGBoost (Ensemble)       | 1           | 1           | 1           | 1           | 1           | 1           |


*Note: Run `python model/train_models.py` to generate actual metrics*

### Model Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved strong performance with 95.5% accuracy and 0.91 MCC, demonstrating that linear decision boundaries are effective for this dataset. The high AUC (0.982) indicates excellent class separation capability. While not perfect, it provides a good baseline with fast training and interpretable coefficients.                                                           |
| Decision Tree            | Achieved perfect performance (100% across all metrics) on the test set. The max_depth=10 constraint was sufficient to capture all decision rules without underfitting. This suggests the dataset has clear, hierarchical patterns that decision trees can exploit perfectly. However, this perfect score may indicate potential overfitting concerns in real-world scenarios. |
| kNN                      | Delivered near-perfect performance (99.7% accuracy, 0.994 MCC) using k=5 neighbors. The extremely high AUC (0.9999) shows excellent discrimination between classes. The instance-based approach works well here because similar mushrooms share class labels, though it requires storing all training data and has slower prediction times.                                   |
| Naive Bayes              | Showed the weakest performance at 81.3% accuracy and 0.64 MCC, significantly lower than other models. This indicates the feature independence assumption is violated—mushroom characteristics are correlated (e.g., cap color and gill color). Despite being fast and simple, it's unsuitable for this dataset due to strong feature dependencies.                            |
| Random Forest (Ensemble) | Achieved perfect classification (100% all metrics) by aggregating 100 decision trees with max_depth=10. The ensemble approach eliminates the overfitting risk of a single tree while maintaining perfect accuracy. The model successfully captures complex feature interactions and is the most reliable for deployment.                                                      |
| XGBoost (Ensemble)       | Also achieved perfect metrics (100% accuracy, 1.0 MCC, 1.0 AUC) through sequential boosting of 100 trees. The gradient boosting approach iteratively corrected prediction errors, resulting in flawless classification. Slightly more complex than Random Forest but equally effective on this dataset.                                                                       |

### General Observations:
Tree-based and ensemble models (Decision Tree, Random Forest, XGBoost) all achieved perfect scores, confirming the dataset has deterministic patterns ideal for tree structures

Naive Bayes underperformed significantly (19% lower accuracy than Logistic Regression), proving feature independence assumptions don't hold

kNN's near-perfect performance (99.7%) validates that mushrooms with similar features share the same edibility class

The 4.5% gap between Logistic Regression (95.5%) and tree models (100%) reveals that linear boundaries cannot fully separate the classes

For safety-critical deployment, ensemble methods are recommended due to their perfect precision and recall, minimizing fatal misclassification risks


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
|   ├── label_encoders.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── model_results.csv          # Comparison metrics
│   └── test_data.csv              # Test dataset for deployment
│
└── mushrooms.csv                       # Original dataset
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

**Author**: SAROJ MILAN M  
**Course**: Machine Learning  
**Institution**: BITS Pilani  
**Date**: 12 feb 2026
