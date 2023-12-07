
def train():
    import re
    import string
    import pandas as pd
    from numpy import array
    from nltk.corpus import stopwords
    from keras.preprocessing.text import Tokenizer
    from keras.models import Sequential
    from keras.layers import Dense
    import wandb
    import os
    import nltk
    from wandb.keras import WandbCallback
    from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Bidirectional
    from tensorflow.keras.regularizers import L1, L2, L1L2
    from tensorflow.keras.optimizers import Adam, RMSprop
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    import tensorflow as tf

    # Ensure that NLTK Stopwords are downloaded
    nltk.download('stopwords')

    # Function to load data
    def load_data(data_dir):
        df = pd.read_csv(data_dir)
        return df['text'], array(df['label'])

    # Function to load vocabulary
    def load_vocab(vocab_dir):
        with open(vocab_dir, 'r') as file:
            vocab = file.read().split()
        return set(vocab)

    # Function to clean the documents
    def clean_doc(doc):
        tokens = doc.split()
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [re_punc.sub('', w) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if len(word) > 1]
        return tokens

    # Function to filter documents by vocabulary
    def filter_by_vocab(docs, vocab):
        new_docs = []
        for doc in docs:
            tokens = clean_doc(doc)
            tokens = [w for w in tokens if w in vocab]
            new_docs.append(' '.join(tokens))
        return new_docs

    # Function to create the tokenizer
    def create_tokenizer(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    # Function to define the model
    def define_model(n_words):
        model = Sequential()
        model.add(Dense(50, input_shape=(n_words,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Function to build a multi-layer deep text classification model
    def build_multi_layer_model(n_words):
        model = Sequential()
        model.add(Dense(128, input_shape=(n_words,), activation='relu', kernel_regularizer=L1(0.0005)))
        model.add(Dense(64, activation='relu', kernel_regularizer=L1L2(0.0005)))
        model.add(Dense(32, activation='relu', kernel_regularizer=L2(0.0005)))
        model.add(Dense(16, activation='relu', kernel_regularizer=L2(0.0005)))
        model.add(Dense(8, activation='relu', kernel_regularizer=L2(0.0005)))
        model.add(Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # Function to build a Multilayer Bidirectional LSTM Model
    def build_bidirectional_lstm_model(n_words):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(1,n_words)))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(64, activation='elu', kernel_regularizer=L1L2(0.0001)))
        model.add(Dense(32, activation='elu', kernel_regularizer=L2(0.0001)))
        model.add(Dense(8, activation='elu', kernel_regularizer=L2(0.0005)))
        model.add(Dense(8, activation='elu'))
        model.add(Dense(4, activation='elu'))
        model.add(Dense(1, activation='sigmoid'))
        opt = RMSprop(learning_rate=0.0001, rho=0.8, momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # Function to build a Transformer Model
    def build_transformer_model():
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model, tokenizer

    # Initialize wandb run for training
    wandb.init(project="sentiment_analysis", job_type="train_all_models")

    # Use W&B artifact for training data
    train_data_artifact = wandb.use_artifact('sentiment_analysis/train_data:v0', type='TrainData')
    train_data_dir = train_data_artifact.download()

    # Use W&B artifact for test data
    test_data_artifact = wandb.use_artifact('sentiment_analysis/test_data:v0', type='TestData')
    test_data_dir = test_data_artifact.download()

    # Use W&B artifact for vocabulary
    vocab_artifact = wandb.use_artifact('sentiment_analysis/vocab:v0', type='Vocab')
    vocab_dir = vocab_artifact.download()

    # Load the vocabulary
    full_vocab_dir = os.path.join(vocab_dir, 'vocabulary.txt')
    vocab = load_vocab(full_vocab_dir)

    # Load all reviews
    # Train
    full_train_data_dir = os.path.join(train_data_dir, 'train_data.csv')
    train_docs, y_train = load_data(full_train_data_dir)
    train_docs = filter_by_vocab(train_docs, vocab)

    # Create the tokenizer
    tokenizer_train = create_tokenizer(train_docs)

    # Encode data
    x_train = tokenizer_train.texts_to_matrix(train_docs, mode='freq')

    # Validation
    full_test_data_dir = os.path.join(test_data_dir, 'test_data.csv')
    test_docs, y_test = load_data(full_test_data_dir)
    test_docs = filter_by_vocab(test_docs, vocab)

    # Encode data
    x_test = tokenizer_train.texts_to_matrix(test_docs, mode='freq')

    # Define the Shallow Neural Network model
    n_words_shallow = x_train.shape[1]
    model_shallow = define_model(n_words_shallow)

    # Fit Shallow Neural Network
    model_shallow.fit(x_train,
                      y_train,
                      epochs=10,
                      verbose=0,
                      validation_data=(x_test, y_test),
                      callbacks=[wandb.keras.WandbCallback(save_model=True, compute_flops=True)])

    # Define the Multi-layer deep text classification model
    n_words_multi_layer = x_train.shape[1]
    model_multi_layer = build_multi_layer_model(n_words_multi_layer)

    # Fit Multi-layer deep text classification model
    model_multi_layer.fit(x_train, y_train, epochs=10, verbose=2, validation_data=(x_test, y_test), callbacks=[WandbCallback(save_model=True, compute_flops=True)])

    # Define the Multilayer Bidirectional LSTM Model
    n_words_lstm = x_train.shape[1]
    model_lstm = build_bidirectional_lstm_model(n_words_lstm)

    # Reshape data for LSTM
    x_train_lstm = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_lstm = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # Fit Multilayer Bidirectional LSTM Model
    model_lstm.fit(x_train_lstm, y_train, epochs=10, verbose=2, validation_data=(x_test_lstm, y_test), callbacks=[WandbCallback(save_model=True, compute_flops=True)])

    # Define the Transformer Model
    model_transformer, tokenizer_transformer = build_transformer_model()

    # Tokenize the text data
    train_encodings = tokenizer_transformer(list(train_docs), truncation=True, padding=True)
    test_encodings = tokenizer_transformer(list(test_docs), truncation=True, padding=True)

    # Create TensorFlow datasets
    train_dataset_transformer = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        tf.constant(y_train, dtype=tf.int32)
    ))

    test_dataset_transformer = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        tf.constant(y_test, dtype=tf.int32)
    ))

    train_dataset_transformer = train_dataset_transformer.batch(16)
    test_dataset_transformer = test_dataset_transformer.batch(16)

    # Fit Transformer Model
    model_transformer.fit(train_dataset_transformer, epochs=1, validation_data=train_dataset_transformer)

    # Finish the W&B run
    wandb.finish()
