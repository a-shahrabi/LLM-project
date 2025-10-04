# Twitter Emotion Classification using BERT

Classify short tweets into six emotions using a fine-tuned BERT model:

- sadness  
- joy  
- love  
- anger  
- fear  
- surprise

---

## Features
- BERT fine-tuning with `Trainer`
- Stratified train/validation split
- Weighted & macro F1 (plus accuracy)
- Early stopping (best model restored)
- Dynamic padding via `DataCollatorWithPadding`
- Reproducible runs (`set_seed`)

## Dataset
Uses https://huggingface.co/datasets/dair-ai/emotion  
Local CSV also supported with columns: `text`, `label` in `{sadness, joy, love, anger, fear, surprise}`.

## Requirements
```bash
pip install torch transformers datasets pandas numpy scikit-learn matplotlib

git clone https://github.com/your-username/bert-twitter-emotion.git
cd bert-twitter-emotion
jupyter notebook notebooks/BERT_Twitter_Emotion_Classification.ipynb
