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

```
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

label_dict = {'sadness':0,'joy':1,'love':2,'anger':3,'fear':4,'surprise':5}
id2label = {v:k for k,v in label_dict.items()}

mdl = AutoModelForSequenceClassification.from_pretrained("models/bert-emotion")
tok = AutoTokenizer.from_pretrained("models/bert-emotion")
def predict(texts):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        probs = mdl(**enc).logits.softmax(-1)
        ids = probs.argmax(-1).tolist()
    return [(t, id2label[i], float(probs[j, i])) for j,(t,i) in enumerate(zip(texts, ids))]```
    

