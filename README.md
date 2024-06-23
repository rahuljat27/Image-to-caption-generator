# Image Captioning with Vision-Encoder-Decoder Model

This project demonstrates an image captioning system that generates descriptive text for images using a combination of vision and language models. Specifically, it leverages a Vision Transformer (ViT) as the encoder and GPT-2 as the decoder to create captions for images. The dataset used for training and evaluation is the COCO2014 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Image captioning is a challenging task that lies at the intersection of computer vision and natural language processing. This project aims to develop a robust image captioning model by combining the strengths of a Vision Transformer (ViT) for image feature extraction and GPT-2 for text generation.

## Model Architecture

The model architecture consists of two main components:

- **Encoder**: A Vision Transformer (ViT) model from Microsoft (`microsoft/swin-base-patch4-window7-224-in22k`) is used to process images and extract their features.
- **Decoder**: GPT-2 (`gpt2`) is used to process the image features and generate descriptive captions.

## Dataset

The COCO2014 dataset, a large-scale object detection, segmentation, and captioning dataset, is used for training and evaluation. It consists of images with corresponding captions.

## Training

The model is trained using the Hugging Face `transformers` library with the following configuration:

- **Batch Size**: 4
- **Epochs**: 1
- **Learning Rate**: Set according to the default `Seq2SeqTrainer` settings
- **Optimizer**: AdamW

During training, the image features extracted by the encoder are fed into the decoder to generate captions. The loss is computed using the cross-entropy loss function, and the model is evaluated using BLEU and ROUGE metrics.

## Evaluation

After training, the model is evaluated on a validation and test set. The evaluation metrics include BLEU and ROUGE scores to assess the quality of the generated captions.

## Results

The model achieved the following results on the test set:

- **ROUGE-1**: 52.7259
- **ROUGE-2**: 23.8143
- **ROUGE-L**: 51.1198
- **BLEU**: 17.1926
- **Average Caption Length**: 11.2 tokens
- **Test Loss**: 0.8015

## Installation

To run this project, you need to install the required packages. You can install them using `pip`:

```bash
pip install torch transformers datasets evaluate tqdm pillow numpy scipy
```


## Usage

To use the model for generating captions, follow these steps:

1. **Load the model**:

    ```python
    from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor
    import torch

    # Load the model
    model = VisionEncoderDecoderModel.from_pretrained("https://drive.google.com/drive/folders/18Kv49GFPfZVYVdV5jPvRHmqLmBqGjsbp?usp=share_link")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Load the image processor
    image_processor = ViTImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    ```

2. **Preprocess the image**:

    ```python
    from PIL import Image

    # Load and preprocess the image
    image = Image.open("path_to_image.jpg")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # Move the pixel values to the same device as the model
    pixel_values = pixel_values.to("cuda" if torch.cuda.is_available() else "cpu")
    ```

3. **Generate the caption**:

    ```python
    # Generate the caption
    outputs = model.generate(pixel_values, max_length=32, num_beams=4)
    
    # Decode the generated caption
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Caption:", caption)
    ```

This will load your trained model, preprocess an input image, and generate a caption for the image.
