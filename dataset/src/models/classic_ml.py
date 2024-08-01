
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CLASSIC_ML_MODELS = {
    "Logistic Regression": (LogisticRegression, {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [5000],
    }),
    "Random Forest": (RandomForestClassifier, {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
    }),
#     "KNN": (KNeighborsClassifier, {
#         "n_neighbors": [3, 5, 7, 10], 
#         "weights": ["uniform", "distance"], 
#         "metric": ["euclidean", "manhattan"]
#     }),
    "NaiveBayes": (GaussianNB, {
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    }),
    "AdaBoost": (AdaBoostClassifier, {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1]
    }),
#     "MLP": (MLPClassifier, {
#         'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
#         'alpha': [0.001, 0.01],
#         'max_iter': [5000]
#     }),
    "SVM": (SVC, {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'probability': [True]
    }),
    "GBDT": (GradientBoostingClassifier, {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [1, 3, 5]
    }),
    "XGBoost": (XGBClassifier, {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    # "LightGBM": (LGBMClassifier, {
    #     'n_estimators': [50, 100, 150],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'num_leaves': [31, 50],
    #     'force_col_wise': [True],
    #     'verbose': [-1]
    # }),
#     'GPC_RBF': (GaussianProcessClassifier, {
#         'kernel': [1.0 * RBF(length_scale=1.0)], 
#         'max_iter_predict': [100], 
#         'n_restarts_optimizer': [0, 1, 2]
#     }),
#     'GPC_Matern': (GaussianProcessClassifier, {
#         'kernel': [1.0 * Matern(length_scale=1.0, nu=1.5)], 
#         'max_iter_predict': [100], 
#         'n_restarts_optimizer': [0, 1, 2]
#     })
}
    

def get_classic_ml_models():
    return CLASSIC_ML_MODELS
    