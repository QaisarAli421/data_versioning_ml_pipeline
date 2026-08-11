import numpy as np
import pandas as pd
import pickle
import json
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


# logging configuration

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# load model

try:
    clf = pickle.load(open('model.pkl', 'rb'))
    logger.debug('Model loaded successfully')

except FileNotFoundError as e:
    logger.error(f'Model file not found: {e}')
    raise

except Exception as e:
    logger.error(f'Error loading model: {e}')
    raise


# fetch the test data

try:
    test_data = pd.read_csv('./data/features/test_bow.csv')
    logger.debug('Test data loaded successfully')

except FileNotFoundError as e:
    logger.error(f'Test data file not found: {e}')
    raise

except pd.errors.ParserError as e:
    logger.error(f'Error parsing test CSV file: {e}')
    raise

except Exception as e:
    logger.error(f'Error loading test data: {e}')
    raise


try:
    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values

    logger.debug('Test features and labels extracted successfully')

except Exception as e:
    logger.error(f'Error extracting test features and labels: {e}')
    raise


# make predictions

try:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    logger.debug('Predictions generated successfully')

except Exception as e:
    logger.error(f'Error while generating predictions: {e}')
    raise


# Calculate evaluation metrics

try:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.debug('Evaluation metrics calculated successfully')

except ValueError as e:
    logger.error(f'Value error while calculating metrics: {e}')
    raise

except Exception as e:
    logger.error(f'Error while calculating evaluation metrics: {e}')
    raise


metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}


# save metrics

try:
    with open('metrics.json', 'w') as file:
        json.dump(metrics_dict, file, indent=4)

    logger.debug('Metrics saved successfully to metrics.json')

except PermissionError as e:
    logger.error(f'Permission denied while saving metrics: {e}')
    raise

except Exception as e:
    logger.error(f'Error while saving metrics: {e}')
    raise


logger.debug('Model evaluation completed successfully')