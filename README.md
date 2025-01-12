# Basic Diffusion Model Implementation

A simple implementation of a diffusion model using PyTorch. This model is trained on the CIFAR-10 dataset and can generate new images through the diffusion process.

## Overview

This implementation includes:
- A simple U-Net architecture for noise prediction
- Forward diffusion process
- Reverse diffusion process (sampling)
- Training loop with CIFAR-10 dataset
- Image generation capabilities

## Requirements

```
torch >= 2.0.0
torchvision >= 0.15.0
tqdm
```

You can install the required packages using:

```bash
pip install torch torchvision tqdm
```

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Minimum 8GB RAM
- Around 2GB free disk space (for CIFAR-10 dataset)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sanyammmm/Modernest_DiffusionModel.git
cd Modernest_DiffusionModel
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model and generate samples:

```bash
python Modernest_DiffusionModel.py
```

The script will:
1. Download the CIFAR-10 dataset automatically
2. Train the diffusion model for 5 epochs
3. Generate and save sample images after each epoch

## Model Architecture

The implementation uses a simplified U-Net architecture with:
- 2 encoder blocks with convolution and downsampling
- 2 decoder blocks with transposed convolution and upsampling
- Time embedding added to the input
- Batch normalization and ReLU activation functions

## Generated Samples

Generated samples will be saved as PNG files in your working directory with names like:
- `samples_epoch_1.png`
- `samples_epoch_2.png`
etc.

## Training Parameters

- Number of timesteps: 1000
- Batch size: 64
- Learning rate: 0.001
- Number of epochs: 5
- Beta schedule: Linear from 1e-4 to 0.02

## Directory Structure

```
Modernest_DiffusionModel/
│
├── Modernest_DiffusionModel.py    # Main implementation file
├── requirements.txt            # Package dependencies
├── README.md                   # This file
└── data/                      # CIFAR-10 dataset (downloaded automatically)
```

## Customization

You can modify the following parameters in the code:
- `timesteps` in `DiffusionModel.__init__`
- `n_epochs` in the `main()` function
- Batch size in the DataLoader
- Model architecture in the `SimpleUNet` class
- Learning rate in the optimizer initialization

## Troubleshooting

Common issues and solutions:

1. CUDA out of memory:
   - Reduce batch size
   - Use a smaller model
   - Train on CPU by forcing `device = "cpu"`

2. Slow training:
   - Ensure you're using GPU acceleration
   - Reduce the number of timesteps
   - Reduce the dataset size for testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```
@misc{Modernest_DiffusionModel,
  author = {Modernest},
  title = {Modernest_DiffusionModel Implementation},
  year = {2025},
  publisher = {GitHub},
  url = https://github.com/Sanyammmm/Modernest_DiffusionModel
}
```
