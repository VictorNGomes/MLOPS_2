import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pandas as pd

# Inicializar a execução do WandB
wandb.init(project='sentiment_analysis', save_code=True)

artifact = wandb.use_artifact('disaster_tweets:v1')

# Download the content of the artifact to the local directory
artifact_dir = artifact.download()

df = pd.read_csv('train.csv')

# Drop de colunas desnecessárias
df = df.drop(['id', 'keyword', 'location'], axis=1)
print(df.info())
print(df.head())

# Contagem de valores na coluna 'target'
target_counts = df['target'].value_counts()

# Visualização da contagem de tweets por categoria
sns.countplot(x='target', data=df)
plt.title('Tweet Count by Category')
plt.savefig('tweet_count.png')  # Save the plot to a file
plt.show()
plt.close()

#Log the histogram image to wandb
wandb.log({'Tweet Count by Category': wandb.Image('tweet_count.png')})

# Log da contagem de valores normalizada no WandB
wandb.log({"Normalized Tweet Count by Category": (target_counts / target_counts.sum()).to_dict()})


# Finalizar a execução do WandB (opcional)
wandb.finish()
