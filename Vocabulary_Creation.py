def vocabulary_creation():    
    import wandb
    import pandas as pd
    import string
    import re
    from collections import Counter
    from nltk.corpus import stopwords
    import nltk
    import os

    # Ensure that NLTK Stopwords are downloaded
    nltk.download('stopwords')

    # Function to load text data from a Pandas DataFrame
    def load_data_from_dataframe(df):
        return df['text'].tolist()

    # Function to clean a document and tokenize it
    def clean_doc(doc):
        tokens = doc.split()
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub('', w) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    # Function to add documents to a vocabulary
    def add_docs_to_vocab(texts, vocab):
        for doc in texts:
            tokens = clean_doc(doc)
            vocab.update(tokens)

    # Function to save a list to a file
    def save_list(lines, filename):
        data = '\n'.join(lines)
        with open(filename, 'w') as file:
            file.write(data)

    # Function to generate and log vocabulary
    def generate_and_log_vocab(train_data_df, run):
        # Load text data
        texts = load_data_from_dataframe(train_data_df)

        # Define vocab
        vocab = Counter()

        # Add all docs to vocab
        add_docs_to_vocab(texts, vocab)

        # Log the size of the vocab
        run.log({'initial_vocab_size': len(vocab)})

        # Keep tokens with a min occurrence
        min_occurrence = 2
        tokens = [k for k, c in vocab.items() if c >= min_occurrence]
        run.log({'filtered_vocab_size': len(tokens)})

        # Save tokens to a vocabulary file
        vocab_file_path = 'vocabulary.txt'
        save_list(tokens, vocab_file_path)

        # Create a new artifact for the vocabulary
        vocab_artifact = wandb.Artifact(
            name='vocab',
            type='Vocab',
            description='Vocabulary from training data'
        )

        # Add vocabulary file to the artifact
        vocab_artifact.add_file(vocab_file_path)

        # Log the new artifact to wandb
        run.log_artifact(vocab_artifact)

    # Initialize wandb run
    run = wandb.init(project='sentiment_analysis', job_type='generate_vocab')

    # Download the train_data.csv artifact
    artifact = run.use_artifact('train_data:latest')
    train_data_path = artifact.download()


    # Load the training data
    train_data_df = pd.read_csv('train_data.csv')

    # Generate and log vocabulary
    generate_and_log_vocab(train_data_df, run)

    # Finish the wandb run and upload the artifacts to the cloud
    run.finish()
