from Preprocessing_Data import remove_punctuations_numbers, tokenize_text, remove_stopwords, lemmatize_text


import pandas as pd
import pytest



TEST_DATA_PATH = 'preprocessed_data.csv'



@pytest.fixture
def sample_data():
    return pd.read_csv(TEST_DATA_PATH)

def test_end_to_end_preprocessing(sample_data):
    # Preprocess a single sample text through the entire pipeline
    sample_text = sample_data['text'][0]

    # Simulate the preprocessing pipeline
    processed_text = remove_punctuations_numbers(sample_text)
    processed_tokens = tokenize_text(processed_text)
    processed_tokens_without_stopwords = remove_stopwords(processed_tokens)
    final_processed_text = lemmatize_text(processed_tokens_without_stopwords)
    final_processed_text = " ".join(final_processed_text)

    # Define the expected final output after preprocessing
    expected_final_output = sample_data['final'][0]

    # Assert that the final processed text matches the expected output
    assert final_processed_text == expected_final_output
