"""
Text Classification using BERT
This notebook demonstrates how to use BERT for classifying Twitter messages by emotion
"""

import os
import logging
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, 
    BertModel, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    set_seed
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score, 
    ConfusionMatrixDisplay, 
    confusion_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

def check_gpu() -> torch.device:
    """Check for GPU availability and return appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using CPU instead.')
    return device

def load_and_preprocess_data(
    file_path: str, 
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to the CSV file
        sample_size: Optional limit on dataset size
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, index_col=0)
        logger.info(f"Successfully loaded dataset with {len(df)} messages")
        
        # Limit dataset size if specified
        if sample_size is not None:
            df = df.iloc[:sample_size]
            logger.info(f"Limited dataset to {len(df)} messages")
        
        # Clean dataset
        missing_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        logger.info(f"Found {missing_count} missing values and {duplicate_count} duplicates")
        
        df = df.dropna()
        df = df.drop_duplicates()
        logger.info(f"After cleaning: {len(df)} messages remain")
        
        # Create emotion mapping
        label_dict = {
            'sadness': 0,
            'joy': 1,
            'love': 2,
            'anger': 3,
            'fear': 4,
            'surprise': 5
        }
        
        # Map string labels to numerical labels
        df['label'] = df['label'].map(label_dict)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading or preprocessing data: {str(e)}")
        raise

def visualize_class_distribution(df: pd.DataFrame, label_names: List[str]) -> None:
    """Visualize the distribution of emotion categories."""
    try:
        plt.figure(figsize=(10, 6))
        df['label'].value_counts().sort_index().plot(kind='bar')
        plt.xticks(range(len(label_names)), label_names, rotation=45)
        plt.xlabel('Emotion Categories')
        plt.ylabel('Frequency')
        plt.title('Distribution of Emotion Categories')
        plt.tight_layout()
        plt.savefig('emotion_distribution.png')  # Save the figure
        plt.show()
        logger.info("Class distribution visualization created")
    except Exception as e:
        logger.error(f"Error visualizing class distribution: {str(e)}")

def tokenize_dataset(
    df: pd.DataFrame, 
    tokenizer: BertTokenizer,
    max_len: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize the text data using BERT tokenizer.
    
    Args:
        df: DataFrame containing the dataset
        tokenizer: BERT tokenizer instance
        max_len: Maximum length for tokenization (if None, calculated automatically)
        
    Returns:
        Tuple of (input_ids, attention_masks, labels)
    """
    try:
        # Find the maximum sentence length if not provided
        if max_len is None:
            max_len = 0
            for text in df['text']:
                tokens = tokenizer.tokenize(text)
                max_len = max(max_len, len(tokens) + 2)  # +2 for [CLS] and [SEP] tokens
            
            # Round to next power of 2 for efficiency
            max_len = 2**int(np.ceil(np.log2(max_len)))
            logger.info(f"Automatically set max_len to {max_len}")
        
        # Tokenize all sentences
        input_ids = []
        attention_masks = []
        
        for text in df['text']:
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
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
        labels = torch.tensor(df['label'].values)
        
        logger.info(f"Tokenization complete - {len(input_ids)} examples processed")
        return input_ids, attention_masks, labels
    
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        raise

def prepare_datasets(
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.1
) -> Tuple[Dataset, Dataset]:
    """
    Split data and prepare datasets for training.
    
    Args:
        input_ids: Tensor of input IDs
        attention_masks: Tensor of attention masks
        labels: Tensor of labels
        test_size: Proportion of data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    try:
        # Split the data into train and validation sets
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            input_ids, labels, random_state=RANDOM_SEED, test_size=test_size, stratify=labels
        )
        
        train_masks, val_masks, _, _ = train_test_split(
            attention_masks, labels, random_state=RANDOM_SEED, test_size=test_size, stratify=labels
        )
        
        # Create DataFrames and convert to Datasets
        train_df = pd.DataFrame({
            'input_ids': train_inputs.tolist(),
            'attention_mask': train_masks.tolist(),
            'label': train_labels.tolist()
        })
        
        val_df = pd.DataFrame({
            'input_ids': val_inputs.tolist(),
            'attention_mask': val_masks.tolist(),
            'label': val_labels.tolist()
        })
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        logger.info(f"Data split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        return train_dataset, val_dataset
    
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        raise

def compute_metrics(eval_pred):
    """Compute metrics for model evaluation."""
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    
    # Calculate metrics
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    
    return {"accuracy": acc, "f1": f1}

def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    output_dir: str = './model_save/'
) -> Trainer:
    """
    Train the BERT model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model: BertForSequenceClassification model
        tokenizer: BERT tokenizer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save the model
        
    Returns:
        Trained Trainer instance
    """
    try:
        # Disable wandb if not needed
        os.environ['WANDB_DISABLED'] = 'true'
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            disable_tqdm=False,
            report_to="none",
            logging_dir=f"{output_dir}/logs",
            logging_steps=len(train_dataset) // batch_size // 4,
            push_to_hub=False,
            log_level="error"
        )
        
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
        logger.info("Starting model training...")
        trainer.train()
        
        # Evaluate the model
        eval_result = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")
        
        return trainer
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def analyze_training_results(trainer: Trainer) -> pd.DataFrame:
    """
    Analyze and visualize training results.
    
    Args:
        trainer: Trained Trainer instance
        
    Returns:
        DataFrame containing metrics for each epoch
    """
    try:
        # Extract training logs
        training_logs = trainer.state.log_history
        
        # Separate train and eval logs
        train_logs = [log for log in training_logs if 'loss' in log and 'eval_loss' not in log]
        eval_logs = [log for log in training_logs if 'eval_loss' in log]
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Epoch': range(1, len(eval_logs) + 1),
            'Training Loss': [log['loss'] for log in train_logs[:len(eval_logs)]],
            'Validation Loss': [log['eval_loss'] for log in eval_logs],
            'Accuracy': [log['eval_accuracy'] for log in eval_logs],
            'F1 Score': [log['eval_f1'] for log in eval_logs]
        })
        
        # Plot training and validation loss
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(metrics_df['Epoch'], metrics_df['Training Loss'], 'b-o', label='Training Loss')
        plt.plot(metrics_df['Epoch'], metrics_df['Validation Loss'], 'r-o', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(metrics_df['Epoch'], metrics_df['Accuracy'], 'g-o', label='Accuracy')
        plt.plot(metrics_df['Epoch'], metrics_df['F1 Score'], 'p-o', label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Accuracy and F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        
        logger.info("Training analysis complete")
        return metrics_df
    
    except Exception as e:
        logger.error(f"Error analyzing training results: {str(e)}")
        raise

def evaluate_model(
    trainer: Trainer,
    val_dataset: Dataset,
    label_names: List[str]
) -> None:
    """
    Evaluate the model and visualize results.
    
    Args:
        trainer: Trained Trainer instance
        val_dataset: Validation dataset
        label_names: List of label names
    """
    try:
        # Get predictions
        predictions = trainer.predict(val_dataset)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        
        # Classification report
        report = classification_report(
            labels, 
            preds, 
            target_names=label_names, 
            output_dict=True
        )
        
        # Print report
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=label_names))
        
        # Plot detailed metrics by class
        metrics_by_class = pd.DataFrame(report).transpose()
        metrics_by_class = metrics_by_class.drop('accuracy', errors='ignore')
        
        plt.figure(figsize=(12, 6))
        metrics_by_class.loc[label_names][['precision', 'recall', 'f1-score']].plot(kind='bar')
        plt.title('Precision, Recall, and F1-Score by Emotion Class')
        plt.xlabel('Emotion')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig('class_metrics.png')
        plt.show()
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        logger.info("Model evaluation and visualization complete")
    
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    output_dir: str
) -> None:
    """
    Save the trained model and tokenizer.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Directory to save the model
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Saving model to {output_dir}")
        
        # Save model and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Log model statistics
        logger.info("Model saved successfully")
        
        # List all files in the model directory
        print("\nFiles in model directory:")
        total_size_mb = 0
        for file in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, file)):
                file_path = os.path.join(output_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
                total_size_mb += size_mb
                print(f"{file}: {size_mb:.2f} MB")
        
        print(f"\nTotal model size: {total_size_mb:.2f} MB")
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def predict_emotion(
    text: str,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    max_len: int,
    device: torch.device,
    label_names: List[str]
) -> Tuple[str, np.ndarray]:
    """
    Predict emotion for a given text.
    
    Args:
        text: Input text
        model: Trained model
        tokenizer: Tokenizer
        max_len: Maximum sequence length
        device: Device to run inference on
        label_names: List of label names
        
    Returns:
        Tuple of (predicted_emotion, confidence_scores)
    """
    try:
        # Tokenize the text
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Move tensors to the device
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Get the predicted label
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Map back to emotion label
        predicted_emotion = label_names[predicted_class]
        
        return predicted_emotion, probabilities.cpu().numpy()[0]
    
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        raise

def main():
    """Main function to run the entire pipeline."""
    try:
        # Set device
        device = check_gpu()
        
        # Define emotion labels
        label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        # Load and preprocess data
        df = load_and_preprocess_data('Emotions_small.csv', sample_size=1370)
        
        # Visualize class distribution
        visualize_class_distribution(df, label_names)
        
        # Load BERT tokenizer
        logger.info('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        # Tokenize dataset
        input_ids, attention_masks, labels = tokenize_dataset(df, tokenizer)
        max_len = input_ids.shape[1]  # Store max_len for later use
        
        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(input_ids, attention_masks, labels)
        
        # Load the model
        logger.info('Loading BERT model...')
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(label_names),
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)
        
        # Train the model
        trainer = train_model(
            train_dataset,
            val_dataset,
            model,
            tokenizer,
            num_epochs=5,
            batch_size=64,
            learning_rate=2e-5
        )
        
        # Analyze training results
        metrics_df = analyze_training_results(trainer)
        print("\nTraining and validation metrics for each epoch:")
        print(metrics_df)
        
        # Evaluate the model
        evaluate_model(trainer, val_dataset, label_names)
        
        # Save the model
        save_model(model, tokenizer, './model_save/')
        
        # Test the model on new unseen Twitter messages
        test_texts = [
            "If karma does not hit you in the face, I will.",
            "The toys R us advert makes me cry."
        ]
        
        print("\nTesting the model on new messages:")
        for text in test_texts:
            emotion, probabilities = predict_emotion(text, model, tokenizer, max_len, device, label_names)
            print(f"Text: '{text}'")
            print(f"Predicted emotion: {emotion}")
            
            # Print confidence for each class
            print("Confidence scores:")
            for i, label in enumerate(label_names):
                print(f"  {label}: {probabilities[i]:.4f}")
            print()
        
        logger.info("Pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")

if __name__ == "__main__":
    main()