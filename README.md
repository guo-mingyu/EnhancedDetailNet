# Enhanced DetailNet

## Introduction

This repository contains the implementation of the EnhancedDetailNet model for image processing tasks. EnhancedDetailNet is a deep learning model designed to capture fine-grained details in images. This README provides an overview of the model, instructions for installation, and guidelines for training.

EnhancedDetailNet has the following key features:
- Preserves the aspect ratio of input images.
- Employs a multi-scale attention mechanism to highlight crucial regions in the image.
- Utilizes global average pooling for comprehensive feature extraction.
- Supports classification tasks.

![EnhancedDetailNet](/model/doc/figures/Slide1.jpg)

## Installation

You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Training
To train the EnhancedDetailNet model on your dataset, follow these steps:

Dataset Preparation: Prepare your dataset in the desired format. Ensure it is organized appropriately for model training.

Configuration: Adjust the model's configuration parameters in the config.yaml file to match your dataset and task requirements.

Training: Run the training script by executing the following command:

## Data Preprocessing

Before training a model on the PlantVillage dataset, it's essential to preprocess the data. The provided code below demonstrates the data preprocessing steps, including data splitting and label encoding.

### Data Splitting

We start by dividing the dataset into three subsets: training, validation, and test. The code selects random images for each category while maintaining an 80-10-10 split ratio.

- 80% of the data is used for training.
- 10% is allocated for validation.
- The remaining 10% is set aside for testing.

```python
# Data Splitting Example
# Code to split the data into train, validation, and test sets
# data_annotator_class.py
# ...

# Training Data File (train_class.txt)
# Validation Data File (val_class.txt)
# Test Data File (test_class.txt)
# ...



## Advanced-300 Channel Mode

The `advanced-300` channel mode in this model represents the highest channel configuration, offering a powerful and detailed feature extraction capability. When using this mode, the model is configured with 300 convolutional channels. This configuration is ideal for tasks that require a comprehensive analysis of fine-grained details within images.

To use the `advanced-300` channel mode in the model, you can set the `channel_mode` parameter in your code as follows:

```python
channel_mode = 'advanced-300'
```

The code does the following:

It organizes the data into training, validation, and test sets.
It shuffles the data to ensure randomness.
It generates files containing image paths and their corresponding class labels.
Usage
Ensure that you have the PlantVillage dataset in the specified data directory.

Use the provided code to perform data classification. Make any necessary adjustments to the data_root variable to specify the location of your dataset.

After running the code, you will have three text files: train_class.txt, val_class.txt, and test_class.txt, which contain the paths to the images and their class labels.

The labels are also saved to a labels_class.txt file.

File Locations
train_class.txt: Paths to images for training.
val_class.txt: Paths to images for validation.
test_class.txt: Paths to images for testing.
labels_class.txt: List of class labels.
You can use these files for training and validation in your machine learning model.

This classification of the data helps ensure that your model is trained on a diverse set of images for optimal performance.
## Parse Arguments

The model includes several command-line arguments that can be configured to tailor the training process to your specific requirements. Here's a brief description of the available arguments:

- `num_classes`: The number of classes in your classification task. The default value is set to 15.

- `input_channels`: The number of input channels in your dataset. The default is 3, which is typical for RGB images.

- `channel_mode`: This argument defines the channel configuration of the model. You can choose from different options, including `lightweight`, `normal`, `advanced`, and `advanced-300`. These modes affect the number of convolutional channels in the model.

- `batch_size`: Determines the batch size used during training. The default value is set to 2, but you can adjust it to optimize training performance based on your hardware resources.

- `learning_rate`: The learning rate for the optimizer during training. The default is set to `1e-6`, but you may need to experiment with different learning rates to achieve optimal training results.

- `num_epochs`: Sets the number of training epochs. The default is 100, but you can modify this value based on the convergence of your model and the complexity of your task.

- `save_interval`: This argument specifies the interval for saving the model during training. You can customize this value based on your preference.

- `input_size`: The size of the input images, which is set to 32 by default. Make sure it matches the resolution of your input data.

You can modify these arguments as needed in your code to adapt the model to your specific task and dataset. Customizing these settings can greatly impact the performance and training process of your model.

## Training the Model

To train the EnhancedDetailNet model, follow these steps:

1. Open a terminal.

2. Navigate to the project directory containing the `train.py` script.

3. Use the following command to initiate the training process:

   ```bash
   python train.py --learning_rate YourSetting --channel_mode advanced-300 --save_interval YourSetting | tee YourSetting.log
   ```
You can modify the command parameters to fine-tune the training process according to your specific requirements. Experiment with different learning rates, channel modes, and other hyperparameters to optimize the model's performance for your task.

## Model Evaluation

The `evaluate.py` script is a valuable tool for assessing the performance of your trained model on the PlantVillage dataset. It enables you to compute various evaluation metrics and analyze the model's effectiveness.

### Usage

You can use `evaluate.py` to evaluate your model's performance by providing the path to your model checkpoint file and the test data. The script will load your model, run it on the test data, and produce evaluation results.

To use `evaluate.py`, you can execute the following command:

```bash
python evaluate.py --model /path/to/your/model.pth --data /path/to/test_class.txt
```


Citation
If you use EnhancedDetailNet in your research, please consider citing our paper:

| Layer (type)               | Output Shape        | Param #        |
|---------------------------|---------------------|----------------|
| Conv2d-1                  | [-1, 300, 32, 32]   | 8,400          |
| ReLU-2                    | [-1, 300, 32, 32]   | 0              |
| Conv2d-3                  | [-1, 300, 32, 32]   | 90,300         |
| ReLU-4                    | [-1, 300, 32, 32]   | 0              |
| Conv2d-5                  | [-1, 300, 32, 32]   | 810,300        |
| Tanh-6                    | [-1, 300, 32, 32]   | 0              |
| Conv2d-7                  | [-1, 300, 32, 32]   | 810,300        |
| Tanh-8                    | [-1, 300, 32, 32]   | 0              |
| Conv2d-9                  | [-1, 300, 32, 32]   | 810,300        |
| Tanh-10                   | [-1, 300, 32, 32]   | 0              |
| Conv2d-11                 | [-1, 300, 32, 32]   | 810,300        |
| Tanh-12                   | [-1, 300, 32, 32]   | 0              |
| ReLU-13                   | [-1, 300, 32, 32]   | 0              |
| Conv2d-14                 | [-1, 18, 32, 32]    | 5,418          |
| Conv2d-15                 | [-1, 18, 32, 32]    | 5,418          |
| Conv2d-16                 | [-1, 300, 32, 32]   | 90,300         |
| SelfAttentionModule-17    | [-1, 300, 32, 32]   | 0              |
| Conv2d-18                 | [-1, 18, 32, 32]    | 5,418          |
| Conv2d-19                 | [-1, 18, 32, 32]    | 5,418          |
| Conv2d-20                 | [-1, 300, 32, 32]   | 90,300         |
| MultiScaleAttentionModule-21 | [-1, 300, 32, 32] | 0              |
| MaxPool2d-22              | [-1, 300, 16, 16]   | 0              |
| Conv2d-23                 | [-1, 600, 16, 16]   | 1,620,600      |
| ReLU-24                   | [-1, 600, 16, 16]   | 0              |
| Conv2d-25                 | [-1, 600, 16, 16]   | 3,240,600      |
| ReLU-26                   | [-1, 600, 16, 16]   | 0              |
| ReLU-27                   | [-1, 600, 16, 16]   | 0              |
| AdaptiveAvgPool2d-28      | [-1, 600, 1, 1]     | 0              |
| Linear-29                 | [-1, 15]            | 9,015          |
| Dropout-30                | [-1, 15]            | 0              |
| Softmax-31                | [-1, 15]            | 0              |

| Params                                 |
|----------------------------------------|
| Total params: 8,412,387                |
| Trainable params: 8,412,387            |
| Non-trainable params: 0                |
|                                        |
| Input size (MB): 0.01                  |
| Forward/backward pass size (MB): 46.86 |
| Params size (MB): 32.09                |
| Estimated Total Size (MB): 78.96       |

[Regarding the above table, a photo illustration of the current model.](/model/doc/figures/MODEL.md)

sql
Copy code
@article{your_article,
  title={A Multi-Scale Attention-Based Model for Image Enhancement and Classification},
  author={guo mingyu},
  journal={},
  year={2023},
}

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Any acknowledgments or credits you want to include, such as related research papers or libraries you've used.
```
This structure provides an overview of your model, instructions for setup and training, guidance for citation, and other essential information. You can customize this template according to your specific model and requirements.
```