# Kidney Disease Detection

## Overview
This project implements a Convolutional Neural Network (CNN) based classifier for detecting kidney disease from CT scan images. The system can identify the presence of kidney stones in medical images, providing a tool for automated diagnosis assistance.

## Project Structure
```
├── .github/workflows/    # GitHub Actions workflows
├── artifacts/            # Model artifacts and processed data
├── cnnClassifier/        # Main package
│   ├── components/       # Model components
│   ├── config/           # Configuration management
│   ├── constants/        # Constants and paths
│   ├── entity/           # Data classes and entities
│   ├── pipeline/         # Training and inference pipelines
│   └── utils/            # Utility functions
├── config/               # Configuration files
├── logs/                 # Application logs
├── Pretrained Models/    # Evaluation notebooks for different models
├── research/             # Research notebooks
└── templates/            # Web application templates
```

## Features
- Data ingestion pipeline for processing CT scan images
- Model training with transfer learning using pre-trained CNN architectures
- Web interface for real-time kidney stone detection
- Comprehensive model evaluation metrics
- DVC integration for experiment tracking

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/manokrishnan123/Kidney_Disease_Detection.git
   cd Kidney_Disease_Detection
   ```

   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

### Training the Model
To train the model from scratch, run:
```
python main.py
```

This will execute the complete pipeline:
1. Data ingestion
2. Base model preparation
3. Model training
4. Model evaluation

### Web Interface
To start the web application for kidney stone detection:
```
python app.py
```

### Making Predictions
1. Upload a CT scan image through the web interface
2. Click the "Predict" button
3. View the prediction results

## Model Architecture
The project implements and compares multiple CNN architectures for kidney stone detection:
- Custom CNN
- VGG16
- ResNet50V2
- MobileNetV3

These models are fine-tuned on kidney stone CT scan images using transfer learning.

## Dataset
The model is trained on a dataset of kidney CT scan images, categorized into:
- Normal kidney images
- Kidney stone images

The dataset is automatically downloaded from GDrive and processed during the data ingestion stage.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
- [Manokrishnan](https://github.com/manokrishnan123)
- Email: manobk08@gmail.com