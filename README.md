# Custom Neural Network for Parking Spot Detection

A deep learning project that implements a **custom Convolutional Neural Network (CNN)** built from scratch using Keras/TensorFlow to detect and classify parking spots as **occupied** or **free** from overhead/camera images. The entire pipeline — from data preprocessing and model architecture design, to training, evaluation, and inference — is documented in a single interactive Jupyter Notebook.

---

## Project Structure

```
Custom-Neural-Network-for-Parking-Spot-Detection/
├── main.ipynb          # Full pipeline: data loading, model definition, training, and evaluation
├── best_model.keras    # Saved weights of the best-performing model checkpoint
└── requirements.txt    # Python dependencies
```

---

## Features

- **Custom CNN Architecture**: A neural network built from scratch using Keras — no pretrained backbone — designed specifically for parking spot binary classification.
- **Binary Classification**: Classifies each parking spot region as either *occupied* or *free*.
- **Model Checkpointing**: Automatically saves the best-performing model during training (`best_model.keras`) using Keras callbacks.
- **Training & Evaluation Pipeline**: Covers the full ML workflow — data loading, augmentation, model compilation, training with validation, and performance metrics — all within a single notebook.
- **Visualization**: Plots training/validation accuracy and loss curves to analyze model convergence and detect overfitting.
- **Ready-to-Use Inference**: The saved `.keras` model can be loaded directly for inference on new parking images without retraining.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Madalin2605/Custom-Neural-Network-for-Parking-Spot-Detection.git
cd Custom-Neural-Network-for-Parking-Spot-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

The notebook expects a labeled dataset of parking spot images (one image per spot, organized by class). Place your dataset in the appropriate directory as referenced inside `main.ipynb`. Common public datasets for this task include [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) and [CNRPark](http://cnrpark.it/).

---

## Usage

### Run the notebook

```bash
jupyter notebook main.ipynb
```

Execute the cells in order to:
1. Load and preprocess the parking spot image dataset.
2. Define and compile the custom CNN architecture.
3. Train the model (best weights are saved to `best_model.keras`).
4. Evaluate performance on the validation/test set.
5. Visualize training curves and classification results.

### Load the saved model for inference

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model('best_model.keras')

# Preprocess your image and run prediction
img = keras.utils.load_img('spot.jpg', target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
img_array = keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
label = "Occupied" if prediction[0][0] > 0.5 else "Free"
print(label)
```

---

## Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras** — model definition, training, and checkpointing
- **NumPy** — numerical operations and array manipulation
- **Matplotlib** — training curve visualization
- **Jupyter Notebook** — interactive development and documentation
- **scikit-learn** *(optional)* — evaluation metrics (confusion matrix, classification report)
