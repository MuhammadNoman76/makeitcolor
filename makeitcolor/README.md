---
license: apache-2.0
---
# MakeItColor: Image Colorization Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10raIuCBUhKCPqIuL_HiSQmkJJ9jbu2VC?usp=sharing)

## Overview

**MakeItColor** is a deep learning model designed for automatic image colorization. It transforms grayscale images into vivid, realistic colorized outputs using a PyTorch-based Convolutional Neural Network (CNN) architecture integrated with the ModelScope framework.

This model builds upon the work of [DDColor](https://github.com/piddnad/DDColor), utilizing a dual-encoder approach and trained on the **ImageNet-Val5k** dataset.

## Features

- **Task**: Image Colorization
- **Framework**: PyTorch, ModelScope
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Grayscale images (single-channel)
- **Output**: Colorized images (RGB format)

## Installation

Ensure you have **Python 3.7+** installed. Then, install the required dependencies:

```bash
pip install opencv-python
pip install modelscope==1.12.0 
pip install datasets==2.14.7
pip install pillow
pip install numpy
```

## Usage

### ModelScope Pipeline

```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from huggingface_hub import snapshot_download

# Download the model files to a local directory
snapshot_download(repo_id="muhammadnoman76/makeitcolor", local_dir="./makeitcolor", repo_type="model")

# Initialize the colorization pipeline
img_colorization = pipeline(Tasks.image_colorization, model='./makeitcolor')

# Load a grayscale image
img_path = 'input.jpg'

# Run colorization
result = img_colorization(img_path)

# Save the colorized image
cv2.imwrite('result.png', result['output_img'])
```

> **Note**: 
> - Ensure that the input image (`input.jpg`) is a proper grayscale (single-channel) image.
> - The output (`result.png`) will be a standard RGB image.

## Google Colab

For an interactive demonstration, try our [Google Colab notebook](https://colab.research.google.com/drive/10raIuCBUhKCPqIuL_HiSQmkJJ9jbu2VC?usp=sharing).

## Model Files

The repository contains:
- `pytorch_model.pt`: Pre-trained model weights
- `configuration.json`: Model configuration file for ModelScope integration
- `README.md`: Documentation

## Requirements

### Hardware
- CPU (supported)
- GPU (recommended for faster inference)

### Software Dependencies
- `modelscope`
- `opencv-python`
- `torch`

## Input Format

- Grayscale images (`.png`, `.jpg`, etc.)

### Example Workflow

1. Prepare a grayscale image (e.g., `input.jpg`)
2. Run the provided example code
3. Check the output file (`result.png`) for the colorized result

## Limitations

- May struggle with highly complex, ambiguous, or abstract grayscale images
- Performance and output quality depend on the clarity and details of the input
- Primarily optimized for natural images; results may vary for synthetic or artistic inputs

## Credits

This work builds upon and was inspired by the [DDColor project](https://github.com/piddnad/DDColor).  
**MakeItColor** leverages a dual-encoder strategy from DDColor and is trained on the **ImageNet-Val5k** dataset.

Special thanks to the creators of DDColor for their foundational contributions.

## License

This project is licensed under the **Apache License 2.0**.

## Contact

For issues, questions, or feedback:
- Open an issue on the Hugging Face repository
- Contact the maintainer directly at: [muhammadnomanshafiq76@gmail.com](mailto:muhammadnomanshafiq76@gmail.com)

---

**Developed by Muhammad Noman**