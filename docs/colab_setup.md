# Google Colab Setup

## 1. Clone repository

```python
!git clone https://github.com/DEIN-USERNAME/gender-bias-hate-speech.git
%cd gender-bias-hate-speech
```

## 2. Install dependencies

```python
!pip install -r requirements.txt
```

## 3. Check GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

## 4. Import project modules

```python
from src.models.model import load_tokenizer, load_classifier
from src.training.train import build_training_args, build_trainer
```

## 5. Keep large artifacts outside Git

Recommended candidates:
- large original datasets
- tokenized datasets if large
- model checkpoints
- full Hugging Face model directories
- large generated logs

Use a persistent external storage location such as Google Drive for these
artifacts when appropriate, and document the path/version used for each run.
