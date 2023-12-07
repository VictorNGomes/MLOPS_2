
def data_segregation():    
    import os
    import wandb
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from Preprocessing_Data import (
        remove_punctuations_numbers,
        tokenize_text,
        remove_stopwords,
        lemmatize_text,
    )

    # Initialize wandb run
    run = wandb.init(project='sentiment_analysis', job_type='data_segregation')

    # Get the preprocessed_data artifact
    artifact = run.use_artifact('preprocessed_data:latest')

    # Download the content of the artifact to the local directory
    artifact.download()

    # Load data from CSV
    df = pd.read_csv('preprocessed_data.csv')

    x = df['final']
    y = df['target']

    # Function to split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Convert split data to DataFrames
    train_data = pd.DataFrame({'text': x_train, 'label': y_train})
    test_data = pd.DataFrame({'text': x_test, 'label': y_test})

    # Log the shapes of the training and testing datasets
    wandb.log({'train_data_shape': train_data.shape, 'test_data_shape': test_data.shape})

    # Save split data to CSV files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    # Create new artifacts for train and test data
    train_artifact = wandb.Artifact(
        name='train_data',
        type='TrainData',
        description='Training data split from preprocessed_data'
    )
    test_artifact = wandb.Artifact(
        name='test_data',
        type='TestData',
        description='Testing data split from preprocessed_data'
    )

    # Add CSV files to the artifacts
    train_artifact.add_file('train_data.csv')
    test_artifact.add_file('test_data.csv')

    # Log the new artifacts to wandb
    run.log_artifact(train_artifact)
    run.log_artifact(test_artifact)

    # Finish the wandb run
    wandb.finish()
