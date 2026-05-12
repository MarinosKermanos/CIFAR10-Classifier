# CIFAR-10 Image Classifier

A PyTorch CNN trained on CIFAR-10 achieving >75% test accuracy.

## Architecture

3-block CNN with BatchNorm and Dropout:

Input (3×32×32)
→ Conv(32) + BN + ReLU + MaxPool → 32×16×16
→ Conv(64) + BN + ReLU + MaxPool → 64×8×8
→ Conv(128) + BN + ReLU + MaxPool → 128×4×4
→ Flatten → FC(512) + Dropout(0.5) → FC(10)

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | >75% |
| Epochs | 30 |
| Optimizer | Adam (lr=0.001) |
| Scheduler | StepLR (step=10, γ=0.5) |

## Training Curves
![curves](models/training_curves.png)

## Confusion Matrix
![cm](models/confusion_matrix.png)

## Setup

```bash
conda activate env (your env)
pip install torch torchvision seaborn scikit-learn
python train.py    # trains and saves best checkpoint
python evaluate.py # generates all plots and metrics
```

## Key Design Choices

- **BatchNorm** after every conv —> stable training
- **Data augmentation** (flip + crop) —> reduces overfitting
- **Train/Val/Test split** —> 45k / 5k / 10k
- **MPS acceleration** —> Apple Silicon GPU support