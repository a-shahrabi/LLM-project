# Twitter Emotion Classification using BERT

This project implements a text classification model using **BERT** to classify Twitter messages into six emotion categories:

- 😢 Sadness  
- 😄 Joy  
- ❤️ Love  
- 😡 Anger  
- 😨 Fear  
- 😲 Surprise  

## 🚀 Project Overview

The goal of this project is to explore **emotion classification** in short texts using the **BERT** model from Hugging Face's `transformers` library. The model is trained on a labeled dataset of tweets to detect the underlying emotion in each message.

We use the [Emotions Dataset](https://huggingface.co/datasets/dair-ai/emotion) of Twitter messages for training and evaluation.

## 🛠️ Requirements

Make sure the following libraries are installed:

- Python 3.6+
- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `datasets`

You can install them using:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib datasets
```

## 📁 Project Structure

```
├── data/               # Dataset files (not included in repo)
├── models/             # Trained model checkpoints
├── notebooks/
│   └── BERT_Twitter_Emotion_Classification.ipynb  # Main notebook
├── README.md
```

## 🧪 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bert-twitter-emotion.git
   cd bert-twitter-emotion
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/BERT_Twitter_Emotion_Classification.ipynb
   ```

3. Follow the steps in the notebook to:
   - Preprocess the dataset
   - Tokenize inputs
   - Fine-tune BERT
   - Evaluate the model

## 📊 Results

The fine-tuned BERT model achieved:

- **Accuracy:** _[insert your accuracy]_
- **F1 Score:** _[insert your F1 score]_