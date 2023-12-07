
from Preprocessing_Data import remove_punctuations_numbers, tokenize_text, remove_stopwords, lemmatize_text


import pandas as pd
import pytest



TEST_DATA_PATH = 'preprocessed_data.csv'



@pytest.fixture
def sample_data():
    return pd.read_csv(TEST_DATA_PATH)



def test_tokenize_text(sample_data):
    processed_text = sample_data['text'][0]  # Replace with the output from the previous step
    processed_tokens = tokenize_text(processed_text)
    expected_output = sample_data['text_tokenized'][0] # Replace with the expected output after tokenization
    assert processed_tokens == eval(expected_output)

def test_remove_stopwords(sample_data):
    processed_tokens = sample_data['text_tokenized'][0]   # Replace with the output from the previous step
    tokens_without_stopwords = remove_stopwords(eval(processed_tokens))
    expected_output = sample_data['text_stop'][0] # Replace with the expected output after removing stopwords
    assert tokens_without_stopwords == eval(expected_output)

def test_lemmatize_text(sample_data):
    tokens_without_stopwords = sample_data['text_stop'][0]  # Replace with the output from the previous step
    final_processed_text = lemmatize_text(eval(tokens_without_stopwords))
    expected_output = sample_data['text_lemmatized'][0]  # Replace with the expected output after lemmatization
    assert final_processed_text == eval(expected_output)

def test_end_to_end_preprocessing(sample_data):
    # Combine the above tests to simulate the end-to-end preprocessing pipeline
    sample_text = sample_data['text'][0]
    
    processed_text = remove_punctuations_numbers(sample_text)
    processed_tokens = tokenize_text(processed_text)
    tokens_without_stopwords = remove_stopwords(processed_tokens)
    final_processed_text = lemmatize_text(tokens_without_stopwords)
    
    final_processed_text = " ".join(final_processed_text)
    expected_final_output = sample_data['final'][0]
    
    assert final_processed_text == expected_final_output
