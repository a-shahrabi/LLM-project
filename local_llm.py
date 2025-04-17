# Text Classification using BERT
# This notebook demonstrates how to use BERT for classifying Twitter messages by emotion

# Check for GPU availability
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

# Loading the Emotions Dataset and Preprocessing it
# Download the dataset
wget.download('https://saref.github.io/teaching/MIE1626/Emotions_small.csv')

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

print(f"\nLength of character dictionary: {len(char_vocab)}")

# Convert the first message to numerical format using the dictionary
char_ids = [char_vocab[char] for char in char_tokens]
print("\nNumerical representation of the first message:")
print(char_ids)

# Convert to one-hot vectors
one_hot = F.one_hot(torch.tensor(char_ids), num_classes=len(char_vocab))
print("\nShape of the one-hot vector representing the message:")
print(one_hot.shape)

# Word tokenization
word_tokens = first_message.split()
print("\nWord tokenized list:")
print(word_tokens)
print(f"Number of word tokens: {len(word_tokens)}")

print("\nDrawbacks of Character and Word Tokenization:")
print("Character Tokenization:")
print("- Results in very long sequences")
print("- Loses semantic meaning of words")
print("- Increases computational complexity")
print("- Poor handling of out-of-vocabulary words")

print("\nWord Tokenization:")
print("- Cannot handle out-of-vocabulary words effectively")
print("- Requires large vocabulary to cover most words")
print("- Loses subword information (e.g., prefixes, suffixes)")
print("- Cannot handle spelling variations well")

# BERT Tokenizer
from transformers import BertTokenizer
# Load the BERT tokenizer
print('\nLoading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Report tokenizer properties
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Maximum length for inputs: {tokenizer.model_max_length}")
print(f"Expected model inputs: {tokenizer.model_input_names}")

# Find the maximum sentence length in the dataset
max_len = 0
for text in df['text']:
    tokens = tokenizer.tokenize(text)
    max_len = max(max_len, len(tokens) + 2)  # +2 for [CLS] and [SEP] tokens

print(f"Maximum sentence length in the dataset (including [CLS] and [SEP]): {max_len}")

# Determine MAX_LEN (next highest power of 2)
MAX_LEN = 2**int(np.ceil(np.log2(max_len)))
print(f"MAX_LEN (next highest power of 2): {MAX_LEN}")

# Encode all sentences in the dataset
input_ids = []
attention_masks = []

for text in df['text']:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Print the 300th Twitter message and its corresponding token ids and attention mask
print("\n300th Twitter message:")
print(df['text'].iloc[299])
print("\nToken IDs:")
print(input_ids[299])
print("\nAttention Mask:")
print(attention_masks[299])

# Decode the tokens to see what they represent
print("\nDecoded tokens:")
print(tokenizer.convert_ids_to_tokens(input_ids[299]))

# Feeding the Data into BERT
# Training and Validation Split
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Convert labels to tensor
labels = torch.tensor(df['label'].values)

# Split the input_ids and labels into train and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, labels, random_state=41, test_size=0.1
)

# Split the attention_masks into train and validation sets
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks, labels, random_state=41, test_size=0.1
)

# Create a DataFrame with the training data
train_df = pd.DataFrame({
    'input_ids': train_inputs.tolist(),
    'attention_mask': train_masks.tolist(),
    'label': train_labels.tolist()
})

# Convert train_df to a Dataset
train_dataset = Dataset.from_pandas(train_df)

# Create a DataFrame with the validation data
val_df = pd.DataFrame({
    'input_ids': validation_inputs.tolist(),
    'attention_mask': validation_masks.tolist(),
    'label': validation_labels.tolist()
})

# Convert val_df to a Dataset
val_dataset = Dataset.from_pandas(val_df)

print("\nTraining dataset info:")
print(train_dataset)
print("\nValidation dataset info:")
print(val_dataset)

# Train the BertForSequenceClassification Model
from transformers import BertForSequenceClassification

# Load the BertForSequenceClassification model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6,
    output_attentions=False,  # Whether the model returns attentions weights
    output_hidden_states=False,  # Whether the model returns all hidden-states
)

# Move model to the device
model.to(device)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    labels = eval_pred.label_ids  # To access the label IDs
    preds_list = eval_pred.predictions  # To access the list of predictions
    preds = preds_list.argmax(-1)  # To decode the predictions using highest value of all classes
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Define the training arguments
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="bert-base-uncased-finetuned-emotion",
    num_train_epochs=5,  # Number of training epochs (adjust as needed)
    learning_rate=2e-5,  # Model learning rate
    per_device_train_batch_size=64,  # Batch size
    per_device_eval_batch_size=64,  # Batch size
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    report_to="none",
    logging_steps=len(train_inputs) // 64,
    push_to_hub=False,
    log_level="error"
)

# Disable wandb
import os
os.environ['WANDB_DISABLED'] = 'true'

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
print("\nTraining the model...")
trainer.train()

# Part 4.3: Model Evaluation
# Extract loss values and metrics from the training log
training_logs = trainer.state.log_history

# Separate train logs and eval logs
train_logs = [log for log in training_logs if 'loss' in log and 'eval_loss' not in log]
eval_logs = [log for log in training_logs if 'eval_loss' in log]

# Create a DataFrame to display the metrics for each epoch
metrics_df = pd.DataFrame({
    'Epoch': range(1, len(eval_logs) + 1),
    'Training Loss': [log['loss'] for log in train_logs],
    'Validation Loss': [log['eval_loss'] for log in eval_logs],
    'Accuracy': [log['eval_accuracy'] for log in eval_logs],
    'F1 Score': [log['eval_f1'] for log in eval_logs]
})

print("\nTraining and validation metrics for each epoch:")
print(metrics_df)

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(metrics_df['Epoch'], metrics_df['Training Loss'], label='Training Loss')
plt.plot(metrics_df['Epoch'], metrics_df['Validation Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Get predictions for the validation dataset
print("\nGenerating predictions for the validation dataset...")
predictions = trainer.predict(val_dataset)

# Decode the predictions
preds = predictions.predictions.argmax(-1)