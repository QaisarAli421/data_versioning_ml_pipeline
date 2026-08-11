import numpy as np
import pandas as pd
import os
import yaml
import logging

from sklearn.feature_extraction.text import CountVectorizer


# logging configuration

logger = logging.getLogger('feature_engineering')
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
    max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']
    logger.debug('max_features retrieved successfully')

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
    logger.error(f'Some error occurred: {e}')
    raise


# fetch the data from data/processed

try:
    train_data = pd.read_csv('./data/processed/train_processed.csv')
    test_data = pd.read_csv('./data/processed/test_processed.csv')
    logger.debug('Training and testing data loaded successfully')

except FileNotFoundError as e:
    logger.error(f'File not found: {e}')
    raise

except pd.errors.ParserError as e:
    logger.error(f'Error parsing CSV file: {e}')
    raise

except Exception as e:
    logger.error(f'Error loading processed data: {e}')
    raise


try:
    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
    logger.debug('Missing values handled successfully')

except Exception as e:
    logger.error(f'Error while filling missing values: {e}')
    raise


# apply BoW

try:
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    logger.debug('Features and labels extracted successfully')

except KeyError as e:
    logger.error(f'Missing column: {e}')
    raise

except Exception as e:
    logger.error(f'Error extracting features and labels: {e}')
    raise


# Apply Bag of Words (CountVectorizer)

try:
    vectorizer = CountVectorizer(max_features=max_features)
    logger.debug('CountVectorizer initialized successfully')

except Exception as e:
    logger.error(f'Error initializing CountVectorizer: {e}')
    raise


# Fit the vectorizer on the training data and transform it

try:
    X_train_bow = vectorizer.fit_transform(X_train)
    logger.debug('Training data transformed successfully')

except Exception as e:
    logger.error(f'Error transforming training data: {e}')
    raise


# Transform the test data using the same vectorizer

try:
    X_test_bow = vectorizer.transform(X_test)
    logger.debug('Test data transformed successfully')

except Exception as e:
    logger.error(f'Error transforming test data: {e}')
    raise


try:
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train

    logger.debug('Training DataFrame created successfully')

except Exception as e:
    logger.error(f'Error creating training DataFrame: {e}')
    raise


try:
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test

    logger.debug('Testing DataFrame created successfully')

except Exception as e:
    logger.error(f'Error creating testing DataFrame: {e}')
    raise


# store the data inside data/features

try:
    data_path = os.path.join("data", "features")

    os.makedirs(data_path, exist_ok=True)

    train_df.to_csv(os.path.join(data_path, "train_bow.csv"))
    test_df.to_csv(os.path.join(data_path, "test_bow.csv"))

    logger.debug('Feature-engineered data saved successfully')

except PermissionError as e:
    logger.error(f'Permission denied while saving data: {e}')
    raise

except Exception as e:
    logger.error(f'Error while saving feature-engineered data: {e}')
    raise


logger.debug('Feature engineering completed successfully')