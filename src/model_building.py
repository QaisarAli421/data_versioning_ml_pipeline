import numpy as np
import pandas as pd
import pickle
import yaml
import logging

from sklearn.ensemble import GradientBoostingClassifier


# logging configuration

logger = logging.getLogger('model_building')
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


# load parameters

try:
    params = yaml.safe_load(
        open('params.yaml', 'r')
    )['model_building']

    logger.debug('Model parameters retrieved successfully')

except FileNotFoundError as e:
    logger.error(f'File not found: {e}')
    raise

except yaml.YAMLError as e:
    logger.error(f'YAML error: {e}')
    raise

except KeyError as e:
    logger.error(f'Missing key in params.yaml: {e}')
    raise

except Exception as e:
    logger.error(f'Some error occurred while loading parameters: {e}')
    raise


# fetch the data from data/features

try:
    train_data = pd.read_csv(
        './data/features/train_bow.csv'
    )

    logger.debug('Training data loaded successfully')

except FileNotFoundError as e:
    logger.error(f'Training data file not found: {e}')
    raise

except pd.errors.ParserError as e:
    logger.error(f'Error parsing training CSV file: {e}')
    raise

except Exception as e:
    logger.error(f'Error loading training data: {e}')
    raise


try:
    X_train = train_data.iloc[:, 0:-1].values
    y_train = train_data.iloc[:, -1].values

    logger.debug('Training features and labels extracted successfully')

except Exception as e:
    logger.error(f'Error extracting training features and labels: {e}')
    raise


# Define and train the Gradient Boosting model

try:
    clf = GradientBoostingClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate']
    )

    logger.debug('Gradient Boosting classifier initialized successfully')

except KeyError as e:
    logger.error(f'Missing model parameter: {e}')
    raise

except Exception as e:
    logger.error(f'Error initializing model: {e}')
    raise


try:
    clf.fit(X_train, y_train)

    logger.debug('Model trained successfully')

except ValueError as e:
    logger.error(f'Value error while training model: {e}')
    raise

except Exception as e:
    logger.error(f'Error while training model: {e}')
    raise


# save

try:
    pickle.dump(
        clf,
        open('model.pkl', 'wb')
    )

    logger.debug('Model saved successfully as model.pkl')

except PermissionError as e:
    logger.error(f'Permission denied while saving model: {e}')
    raise

except Exception as e:
    logger.error(f'Error while saving model: {e}')
    raise


logger.debug('Model building completed successfully')