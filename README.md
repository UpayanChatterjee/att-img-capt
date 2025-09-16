# Image Captioning with Attention

This project implements an image captioning model using an encoder-decoder architecture with an attention mechanism. It provides a comparison between two different image encoders: ResNet50 and a Vision Transformer (ViT). The decoder is an LSTM-based model with Bahdanau attention.

## Features

- Encoder-Decoder architecture for image captioning.
- Attention mechanism to focus on relevant parts of the image.
- Comparison between ResNet50 and Vision Transformer (ViT) as image encoders.
- Training and evaluation pipeline.
- Caption generation for new images.
- Jupyter notebook for demonstration and experimentation.

## Project Structure

```
att-img-capt/
├── data/
│   └── flickr30k/      # Flickr30k dataset images
│   └── results.csv     # Flickr30k captions
├── models/
│   ├── __init__.py
│   ├── decoder.py      # Attention Decoder implementation
│   └── encoders.py     # ResNet50 and ViT Encoder implementations
├── notebooks/
│   └── image_captioning_attention_resnet50_vit.ipynb # Main notebook
├── scripts/
│   ├── data_utils.py   # Utilities for data loading and handling
│   ├── evaluation.py   # Evaluation script (e.g., BLEU score)
│   ├── preprocessing.py # Text and image preprocessing
│   └── train.py        # Training script
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/UpayanChatterjee/att-img-capt.git
    cd att-img-capt
    ```

2.  **Create a Python environment:**
    It is recommended to use a virtual environment.

    ```bash
    python -m venv env
    source env/bin/activate
    ```

3.  **Install dependencies:**
    The project requires the following libraries. You can install them using pip:

    ```bash
    pip install torch torchvision pandas scikit-learn nltk matplotlib Pillow tqdm transformers jupyter
    ```

    You can also create a `requirements.txt` file with the libraries above and install them with `pip install -r requirements.txt`.

4.  **Download NLTK data:**

    ```python
    import nltk
    nltk.download('punkt')
    ```

5.  **Download the dataset:**
    Download the Flickr30k dataset. The images should be placed in `data/flickr30k/` and the captions file (`results.csv`) in `data/`.

## Usage

The primary way to interact with this project is through the Jupyter notebook.

1.  **Start Jupyter:**

    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:**
    Navigate to `notebooks/image_captioning_attention_resnet50_vit.ipynb`.

3.  **Run the cells:**
    Execute the cells in the notebook to train the models, evaluate them, and generate captions. The notebook is self-contained and includes all steps from data loading to visualization.

Alternatively, you can use the scripts in the `scripts/` directory for a more modular approach to training and evaluation.

## Models

### Encoders

Two pre-trained encoders are used to extract features from images:

- **ResNet50:** A convolutional neural network that provides spatial features. The final classification layer is removed.
- **Vision Transformer (ViT):** A transformer-based model that processes images as sequences of patches.

### Decoder

The decoder is an LSTM network with a Bahdanau-style attention mechanism. It takes the encoded image features and generates a caption word by word.
