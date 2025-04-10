# Text Classification using BERT
# This notebook demonstrates how to use BERT for classifying Twitter messages by emotion

#Check for GPU availability
import wget
import gzip, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay, confusion_matrix

# If there's a GPU available...
if torch.cuda.is_available():
    # tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('we will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Loading the Emotions Dataset and Preprocessing it
# Download the dataset
wget.download('Emotions_small.csv')

# Read the CSV file
df = pd.read_csv('Emotions_small.csv', index_col=0)
print(f"Number of messages in the dataset: {len(df)}")

# Display 10 random rows from the DataFrame
print("\nDisplaying 10 random rows from the DataFrame:")
print(df.sample(10))

# Limit the dataset to the first 1370 messages
df = df.iloc[:1370]

# Check for missing values and duplicated rows
print("\nMissing values in the dataset:")
print(df.isnull().sum())

print("\nNumber of duplicated rows:", df.duplicated().sum())

# Remove rows with missing values and duplicated messages
df = df.dropna()
df = df.drop_duplicates()

print("\nNumber of remaining messages after cleaning:", len(df))

# Extract unique labels from the 'label' column
unique_labels = df['label'].unique()
print("\nUnique emotion labels:", unique_labels)

# Create a dictionary to map string labels to numerical labels
label_dict = {
    'sadness': 0,
    'joy': 1,
    'love': 2,
    'anger': 3,
    'fear': 4,
    'surprise': 5
}

# Modify the 'label' column by replacing string labels with numerical labels
df['label'] = df['label'].map(label_dict)

# Visualize class distributions
plt.figure(figsize=(10, 6))
df['label'].value_counts().sort_index().plot(kind='bar')
plt.xticks(range(6), ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], rotation=45)
plt.xlabel('Emotion Categories')
plt.ylabel('Frequency')
plt.title('Distribution of Emotion Categories')
plt.tight_layout()
plt.show()

# Preparing Text Data for BERT

# Character and Word tokenization
# Get the first Twitter message
first_message = df['text'].iloc[0]
print("\nFirst Twitter message:")
print(first_message)

# Character tokenization
char_tokens = list(first_message)
print("\nCharacter tokenized list:")
print(char_tokens)
print(f"Number of character tokens: {len(char_tokens)}")

# Create a dictionary to map each character to a unique integer
char_vocab = {}
for char in char_tokens:
    if char not in char_vocab:
        char_vocab[char] = len(char_vocab)