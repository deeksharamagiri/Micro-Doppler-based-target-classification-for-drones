# Micro-Doppler-based-target-classification-for-drones
UE22CS352A – Algorithms and Optimizations in Machine Learning Course Project 

This project uses deep learning to classify micro-Doppler radar spectrograms to distinguish drones from other objects like birds and clutter. It simulates radar returns for different flying objects under realistic environmental conditions and applies a Convolutional Neural Network (CNN) for multi-class image classification.

---
## Dataset Preparation

### Dataset Description

The dataset consists of synthetic micro-Doppler radar spectrograms simulating radar signatures of various flying objects:

- Birds
- Drones
- Clutter (noise or irrelevant reflections)

Each class has its own subfolder containing grayscale `.png` images representing the radar return over time. These spectrograms are time-frequency representations of motion characteristics.

### Metadata CSV

The dataset is accompanied by a CSV file named `metadata.csv` which contains two columns:

| Column      | Description                                                  |
|-------------|--------------------------------------------------------------|
| `file_path` | Relative path to the image (e.g. `drones/drone_01.png`)      |
| `label`     | Class label (`drones`, `birds`, `clutter`)                   |

**Example:**

```csv
file_path,label
drones/drone_01.png,drones
birds/bird_05.png,birds
clutter/clutter_10.png,clutter
```
## Running the Code

### 1. Install Dependencies

Make sure the required Python packages are installed:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow
```
### 2. Preprocess Data

Run the data loader script to:

- Load spectrogram images from the dataset
- Resize them to `128x128`
- Normalize pixel values to `[0, 1]`
- Encode the labels for training using one-hot encoding

> The processed data will be ready for model training after this step.

### 3. Train the CNN

Use the provided model creation and training code:

```python
model = create_model(input_shape, num_classes)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32
)
model.save("micro_doppler_model.h5")
```
> Training includes EarlyStopping to monitor validation loss and prevent overfitting.

### 4. Evaluate and Predict

Run the evaluation and prediction script to:

- Compute test accuracy and loss  
- Visualize predictions for individual test images  
- Display the confusion matrix for overall performance

```python
model.evaluate(X_test, y_test)
predict_image("/path/to/sample_image.png")
```
> Predictions will be visualized using Matplotlib with the predicted class shown as the title.
