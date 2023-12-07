from Fetch_Data import fetch_data
from EDA import eda
from Preprocessing_Data import preprocessing_data
from Data_Segregation import data_segregation
from Vocabulary_Creation import vocabulary_creation
from Train import train


from pipeline_class import depends_on


@depends_on()
def fetch_data_pipe():
    print("Fetching data...")
    fetch_data()
    

# EDA.py

@depends_on(fetch_data_pipe)
def eda_pipe():
    print("Performing EDA...")
    eda()
   

# Preprocessing_Data.py

@depends_on(eda_pipe)
def preprocessing_data_pipe():
    print("Performing data preprocessing...")
    preprocessing_data()
    

# Data_Check.py

@depends_on(preprocessing_data_pipe)
def data_check_pipe():
    
    print("Checking data...")
    import subprocess

    command = 'pytest Data_Check.py'

    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result)
# Data_Segregation.py

@depends_on(data_check_pipe)
def data_segregation_pipe():
    print("Segregating data...")
    data_segregation()
    

# Vocabulary_Creation.py

@depends_on(data_segregation_pipe)
def vocabulary_creation_pipe():
    print("Creating vocabulary...")
    vocabulary_creation()
    

# Train.py

@depends_on(vocabulary_creation_pipe)
def train_pipe():
    print("Training the model...")
    train()
    

# Run the pipeline
if __name__ == "__main__":
    from pipeline_class import pipeline
    pipeline.run()

