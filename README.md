# az-ner-blur

Azerbaijani Named Entity Recognition (NER) Blur is a machine learning project focused on extracting and anonymizing sensitive entities from Azerbaijani text, such as FIN codes, car plate numbers, and ID numbers. Built using deep learning (transformers, PyTorch, HuggingFace), it offers robust NER for the Azerbaijani language with special focus on privacy and data minimization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bahramzada/az-ner-blur/blob/main/NER_MODEL.ipynb)

## Overview

This repository contains a full pipeline for training and deploying NER models tailored for Azerbaijani-specific entities. It leverages HuggingFace Transformers and PyTorch for model training, evaluation, and inference. The project includes data annotation, advanced preprocessing, model configuration, training logs, and evaluation metrics. The key goal is to automate the anonymization of sensitive text in Azerbaijani documents, supporting privacy-preserving AI solutions.

## About Azerbaijani Text Processing

Azerbaijani language presents unique challenges for NER due to complex morphology, lack of large annotated datasets, and rich entity formats (e.g., FIN codes, car plates). This project tackles these by custom annotation, rule-based preprocessing, and adaptation of transformer models for Azerbaijani language, achieving high accuracy and robust anonymization.

## Features

- ✨ **Sensitive Entity Detection**: Extracts FIN, car plate, and ID numbers from Azerbaijani text.
- ⚡ **Blur/Anonymization Pipeline**: Automated masking of detected entities for privacy.
- 🛠️ **Custom Dataset Creation**: Tools for annotation, conversion to BIO format, and splitting.
- 📈 **Advanced Model Training**: Full training loop, evaluation, and model checkpointing.

## Dataset

- **Source**: Annotated in-house [(example)](https://github.com/bahramzada/az-ner-blur)
- **Size**: ~90,000 samples (Train: 72,000, Validation: 18,000)
- **Language**: Azerbaijani
- **Format**: JSON, CSV (BIO entity format)
- **Classes**: FIN, Car Plate, ID Number, O (other)
- **Preprocessing**: Cleaning, conversion to BIO, rule-based validation, splitting

## Model Details

- **Base Model**: HuggingFace Transformers (e.g., bert-base-cased)
- **Batch Size**: 32
- **Max Sequence Length**: 128
- **Learning Rate**: 2e-5
- **Training Loss**: ~0.0003 (final epoch)
- **Dropout**: 0.1
- **Checkpoints**: `/content/best_model/` (Colab)
- **Total Training Steps**: 300

## Requirements

```
torch>=2.8.0
transformers>=4.41.0
scikit-learn
pandas
numpy
wandb
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bahramzada/az-ner-blur.git
cd az-ner-blur
```

2. Install dependencies:
```bash
pip install torch transformers scikit-learn pandas numpy wandb
```

3. GPU Support (optional):
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Using Google Colab (Recommended)

1. Click the "Open in Colab" badge
2. Run the notebook
3. Follow step-by-step instructions

### Local Usage

1. Open the notebook:
```bash
jupyter notebook NER_MODEL.ipynb
```
2. Follow the steps provided

### Quick Example

```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_path = "/content/best_model"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(model_path)

ner = pipeline("ner", model=model, tokenizer=tokenizer)
text = "AZEDF12 və 10-AB-123 nömrəli maşın."
result = ner(text)
print(result)
```

## Training Process

1. **Data Loading**
2. **Preprocessing**
3. **Tokenization**
4. **Model Training**
5. **Evaluation**

### Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/best_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=3,
    seed=42
)
```

## File Structure

```
az-ner-blur/
├── README.md
├── NER_MODEL.ipynb
├── scripts/           # (if any scripts exist)
└── .git/
```

## Results

- Achieved high accuracy in entity detection (>95% valid detection)
- Robust anonymization of sensitive Azerbaijani entities
- Successfully deployed on Google Colab

## Contributing

We welcome contributions! Open a PR or issue, especially for:
- New entity types
- Dataset expansion
- Improvements to preprocessing or training

## License

This project is open source. See the LICENSE section.

## Acknowledgments

- Dataset annotation contributors
- HuggingFace, PyTorch, WandB libraries
- Data sources: In-house, public examples

## Citation

```bibtex
@misc{az-ner-blur,
  title={Azerbaijani NER Blur},
  author={bahramzada},
  year={2025},
  url={https://github.com/bahramzada/az-ner-blur}
}
```

---

*"Protecting privacy in Azerbaijani language AI — one entity at a time."*
