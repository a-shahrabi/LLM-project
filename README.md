Twitter Emotion Classification using BERT

Classify short tweets into six emotions using a fine-tuned BERT model (Hugging Face Transformers):

sadness

joy

love

anger

fear

surprise

Features

BERT fine-tuning with Trainer

Stratified train/validation split

Weighted F1 and accuracy metrics

Early stopping and best-model restore

Dynamic padding (faster, lower memory)

Reproducible runs with fixed seed

Dataset

This project uses the public Twitter emotions dataset:
https://huggingface.co/datasets/dair-ai/emotion

You may also supply a local CSV with columns:

text — the tweet/message

label — one of {sadness, joy, love, anger, fear, surprise}

Requirements

Python 3.8+

torch

transformers

datasets

pandas

numpy

scikit-learn

matplotlib

Install with:

pip install torch transformers datasets pandas numpy scikit-learn matplotlib


For GPU users, install the CUDA-matched PyTorch wheel from https://pytorch.org/get-started/locally/

Project Structure:

├── data/                          # Optional local data (not committed)
├── models/                        # Saved checkpoints / exported model
├── notebooks/
│   └── BERT_Twitter_Emotion_Classification.ipynb   # Main notebook
├── src/
│   ├── train.py                   # (Optional) script-based training entry
│   ├── infer.py                   # (Optional) simple CLI inference
│   └── utils.py                   # (Optional) helpers: tokenization, metrics, plotting
├── README.md

