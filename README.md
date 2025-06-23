# Vision Transformer Digit Classifier

Train and test a Vision Transformer (ViT) on the MNIST digit images dataset.

## Features

- Vision Transformer (ViT) implementation in PyTorch
- Trains and evaluates on MNIST handwritten digit dataset
- Early stopping and model checkpointing
- Training, validation, and test accuracy logging (with Weights & Biases)
- Easily configurable hyperparameters

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- wandb
- tqdm

Install dependencies:
```
pip install -r requirements.txt
```
Or manually:
```
pip install torch torchvision numpy wandb tqdm
```

## Usage

### Train and Test

Run the main script to train and test the ViT model:
```
python train_test.py
```

This will:
- Train the model on the MNIST training set
- Validate on a held-out validation set
- Test on the MNIST test set
- Log metrics to Weights & Biases (wandb)

### Configuration

You can adjust hyperparameters (batch size, number of layers, heads, epochs, etc.) at the top of `train_test.py`.

## Files

- `vit.py` — Vision Transformer model definition
- `train_test.py` — Training, validation, and testing pipeline
- `README.md` — Project documentation

## Project Structure

```
Vision-Transformer-Digit-Classifier/
├── vit.py
├── train_test.py
├── README.md
└── requirements.txt
```

## References

- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/)
- [Weights & Biases](https://wandb.ai/)

---
