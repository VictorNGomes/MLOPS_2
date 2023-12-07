from Fetch_Data import fetch_data
from EDA import eda
from Preprocessing_Data import preprocessing_data
from Data_Check import data_check
from Data_Segregation import data_segregation
from Vocabulary_Creation import vocabulary_creation
from Train import train


class Pipeline:
    def __init__(self):
        self._functions = []

    def add_function(self, func):
        self._functions.append(func)
        return func

    def run(self):
        for func in self._functions:
            func()

pipeline = Pipeline()

def depends_on(*dependencies):
    def decorator(func):
        func.dependencies = dependencies
        pipeline.add_function(func)
        return func
    return decorator

# Fetch_Data.py

from pipeline import depends_on

@depends_on()
def fetch_data():
    print("Fetching data...")

# EDA.py

@depends_on(fetch_data)
def eda():
    print("Performing EDA...")

# Preprocessing_Data.py

@depends_on(eda)
def preprocessing_data():
    print("Performing data preprocessing...")

# Data_Check.py

@depends_on(preprocessing_data)
def data_check():
    print("Checking data...")

# Data_Segregation.py

@depends_on(data_check)
def data_segregation():
    print("Segregating data...")

# Vocabulary_Creation.py

@depends_on(data_segregation)
def vocabulary_creation():
    print("Creating vocabulary...")

# Train.py

@depends_on(vocabulary_creation)
def train():
    print("Training the model...")

# Run the pipeline
if __name__ == "__main__":
    from pipeline import pipeline
    pipeline.run()

