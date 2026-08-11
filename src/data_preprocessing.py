import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import yaml


# ---------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------------------------------------------------
# FETCH THE DATA FROM DATA/RAW FOLDER
# ---------------------------------------------------------

try:
    train_data = pd.read_csv("./data/raw/train.csv")
    test_data = pd.read_csv("./data/raw/test.csv")

    logger.debug("Training and testing data loaded successfully.")

except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    raise

except pd.errors.EmptyDataError as e:
    logger.error(f"CSV file is empty: {e}")
    raise

except pd.errors.ParserError as e:
    logger.error(f"Error while parsing CSV file: {e}")
    raise

except Exception as e:
    logger.error(f"Unexpected error while loading data: {e}")
    raise


# ---------------------------------------------------------
# DOWNLOAD NLTK DATA
# ---------------------------------------------------------

try:
    nltk.download("wordnet")
    nltk.download("stopwords")

    logger.debug("NLTK resources downloaded successfully.")

except Exception as e:
    logger.error(f"Error while downloading NLTK resources: {e}")
    raise


# ---------------------------------------------------------
# LEMMATIZATION
# ---------------------------------------------------------

def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()

        text = text.split()

        text = [lemmatizer.lemmatize(y) for y in text]

        return " ".join(text)

    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        raise


# ---------------------------------------------------------
# REMOVE STOP WORDS
# ---------------------------------------------------------

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))

        Text = [
            i for i in str(text).split()
            if i not in stop_words
        ]

        return " ".join(Text)

    except Exception as e:
        logger.error(f"Error while removing stop words: {e}")
        raise


# ---------------------------------------------------------
# REMOVE NUMBERS
# ---------------------------------------------------------

def removing_numbers(text):
    try:
        text = "".join([i for i in text if not i.isdigit()])

        return text

    except Exception as e:
        logger.error(f"Error while removing numbers: {e}")
        raise


# ---------------------------------------------------------
# LOWER CASE
# ---------------------------------------------------------

def lower_case(text):
    try:
        text = text.split()

        text = [y.lower() for y in text]

        return " ".join(text)

    except Exception as e:
        logger.error(f"Error while converting text to lowercase: {e}")
        raise


# ---------------------------------------------------------
# REMOVE PUNCTUATIONS
# ---------------------------------------------------------

def removing_punctuations(text):
    try:
        # Remove punctuations
        text = re.sub(
            "[%s]" % re.escape(
                """!"#$%&'()*+,،-./:;<=>؟?@[]^_`{|}~"""
            ),
            " ",
            text
        )

        text = text.replace("؛", "")

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        text = " ".join(text.split())

        return text.strip()

    except Exception as e:
        logger.error(f"Error while removing punctuations: {e}")
        raise


# ---------------------------------------------------------
# REMOVE URLS
# ---------------------------------------------------------

def removing_urls(text):
    try:
        url_pattern = re.compile(
            r"https?://\S+|[www.\S+](http://www.\S+)"
        )

        return url_pattern.sub(r"", text)

    except Exception as e:
        logger.error(f"Error while removing URLs: {e}")
        raise


# ---------------------------------------------------------
# REMOVE SMALL SENTENCES
# ---------------------------------------------------------

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan

        return df

    except Exception as e:
        logger.error(f"Error while removing small sentences: {e}")
        raise


# ---------------------------------------------------------
# NORMALIZE TEXT
# ---------------------------------------------------------

def normalize_text(df):
    try:
        df.content = df.content.apply(
            lambda content: lower_case(content)
        )

        df.content = df.content.apply(
            lambda content: remove_stop_words(content)
        )

        df.content = df.content.apply(
            lambda content: removing_numbers(content)
        )

        df.content = df.content.apply(
            lambda content: removing_punctuations(content)
        )

        df.content = df.content.apply(
            lambda content: removing_urls(content)
        )

        df.content = df.content.apply(
            lambda content: lemmatization(content)
        )

        logger.debug("Text normalization completed successfully.")

        return df

    except KeyError as e:
        logger.error(f"Required column missing during normalization: {e}")
        raise

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# ---------------------------------------------------------
# PROCESS DATA
# ---------------------------------------------------------

try:
    logger.debug("Starting training data preprocessing.")

    train_processed_data = normalize_text(train_data)

    logger.debug("Training data preprocessing completed.")

    logger.debug("Starting testing data preprocessing.")

    test_processed_data = normalize_text(test_data)

    logger.debug("Testing data preprocessing completed.")

except Exception as e:
    logger.error(f"Error during data preprocessing: {e}")
    raise


# ---------------------------------------------------------
# STORE THE DATA INSIDE DATA/PROCESSED
# ---------------------------------------------------------

try:
    data_path = os.path.join("data", "processed")

    os.makedirs(data_path, exist_ok=True)

    train_processed_data.to_csv(
        os.path.join(data_path, "train_processed.csv"),
        index=False
    )

    test_processed_data.to_csv(
        os.path.join(data_path, "test_processed.csv"),
        index=False
    )

    logger.debug("Processed training and testing data saved successfully.")

except PermissionError as e:
    logger.error(f"Permission denied while saving processed data: {e}")
    raise

except Exception as e:
    logger.error(f"Error while saving processed data: {e}")
    raise


logger.debug("Data preprocessing completed successfully.")
