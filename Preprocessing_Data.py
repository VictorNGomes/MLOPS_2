from nltk.corpus import stopwords
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
import wandb
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

lemmatizer = WordNetLemmatizer()
def remove_punctuations_numbers(inputs):
        """Remove punctuations and numbers from the text."""
        return re.sub(r'[^a-zA-Z]', ' ', inputs)
def tokenize_text(inputs):
        """Tokenize the text."""
        return word_tokenize(inputs)

def remove_stopwords(inputs):
        """Remove stopwords from the tokenized text."""
        return [k for k in inputs if k not in stop_words]  

def lemmatize_text(inputs):
        """Lemmatize the text."""
        return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]    

def preprocessing_data():
    import re
    import pandas as pd
    import nltk
    from nltk.tokenize import word_tokenize
    
    from nltk.stem import WordNetLemmatizer
    import wandb
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Initialize wandb run
    wandb.init(project='sentiment_analysis', job_type="preprocessing")

    # Load your DataFrame 'df' here
    artifact = wandb.use_artifact('disaster_tweets:v1')
    artifact_dir = artifact.download()
    df = pd.read_csv('train.csv')


    # Lowercase all the texts
    df['text'] = df['text'].str.lower()

    

    df['text'] = df['text'].apply(remove_punctuations_numbers)

    

    df['text_tokenized'] = df['text'].apply(tokenize_text)

    

    

    df['text_stop'] = df['text_tokenized'].apply(remove_stopwords)

    

    

    df['text_lemmatized'] = df['text_stop'].apply(lemmatize_text)

    # Joining Tokens into Sentences
    df['final'] = df['text_lemmatized'].str.join(' ')

    # Log word clouds as artifacts
    data_disaster = df[df['target'] == 1]
    data_not_disaster = df[df['target'] == 0]

    wordcloud_disaster = WordCloud(max_words=500,
                                random_state=100,
                                background_color='white',
                                collocations=True).generate(str((data_disaster['final'])))

    wordcloud_not_disaster = WordCloud(max_words=500,
                                    random_state=100,
                                    background_color='white',
                                    collocations=True).generate(str((data_not_disaster['final'])))

    wordcloud_disaster_path = 'wordcloud_disaster.png'
    wordcloud_not_disaster_path = 'wordcloud_not_disaster.png'

    wordcloud_disaster.to_file(wordcloud_disaster_path)
    wordcloud_not_disaster.to_file(wordcloud_not_disaster_path)                                   


    # Log the word clouds as artifacts to Weights and Biases
    wordcloud_disaster_artifact = wandb.Artifact(
        name='wordcloud_disaster',
        type='WordCloud'
    )
    wordcloud_disaster_artifact.add_file(wordcloud_disaster_path)
    wandb.log_artifact(wordcloud_disaster_artifact)

    wordcloud_not_disaster_artifact = wandb.Artifact(
        name='wordcloud_not_disaster',
        type='WordCloud'
    )
    wordcloud_not_disaster_artifact.add_file(wordcloud_not_disaster_path)
    wandb.log_artifact(wordcloud_not_disaster_artifact)

    df.to_csv('preprocessed_data.csv', index=False)

    # Log the preprocessed data as an artifact to Weights and Biases
    preprocessed_data_artifact = wandb.Artifact(
        name='preprocessed_data',
        type='PreprocessedData',
        description='DataFrame after text preprocessing'
    )
    preprocessed_data_artifact.add_file('preprocessed_data.csv')
    wandb.log_artifact(preprocessed_data_artifact)

    wandb.finish()