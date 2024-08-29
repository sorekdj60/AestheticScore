
## README.md

# Beauty Score Prediction Using SCUT-FBP5500 Dataset

This project demonstrates how to train a deep learning model to predict facial beauty scores based on the SCUT-FBP5500 V2 dataset. The dataset contains images of Asian and Caucasian male and female faces rated on a scale of 1 to 5 by multiple raters, providing a diverse dataset for exploring the concept of facial attractiveness. 

The objective of this project is to train a model that can predict a beauty score for any given face based on this dataset.

**Note:** This project is intended for learning purposes and does not qualify the physical appearance or inherent worth of any individual.

---

## Dataset Overview

**Dataset:** SCUT-FBP5500 V2 (available on Kaggle: [Link](https://www.kaggle.com/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores/data?select=Images))

- **Images:** 5,500 images in 350x350 resolution.
- **Subjects:** Includes faces of both Asian and Caucasian, male and female subjects.
- **Ratings:** Each face image is rated with a floating-point number ranging from 1.0 to 5.0. 
- **Goal:** To predict the beauty score of a face using a deep learning model trained on these labeled images.

---

## Project Overview

The workflow involves several key stages: data preprocessing, model preparation and training, testing, and beauty score calculation for new images. Below is a breakdown of each step.

### 1. **Data Preparation**

The dataset images are labeled with floating-point beauty scores. The first step is to preprocess the data:

- **Text File Processing:** A text file (`label.txt`) contains the filenames and their corresponding beauty scores. These scores are rounded to the nearest integer to categorize the images into specific classes (1 to 5).
- **Folder Structure:** Images are moved into subfolders based on their rounded score. For example, an image with a score of `3.4` would be rounded to `3` and stored in a folder named `3`.

### 2. **Data Loading and Splitting**

Using PyTorch's `ImageFolder` class, the data is loaded, and transformations such as resizing, normalization, and conversion to tensor format are applied. The dataset is split into training, validation, and test sets using the `random_split` function, with customizable proportions (e.g., 10% for testing and validation).

### 3. **Model Architecture**

The model used for this task is a pre-trained **ResNet-18** model. Pre-trained models are commonly used to leverage the knowledge from large datasets (e.g., ImageNet) and fine-tune them on smaller, specialized datasets like SCUT-FBP5500.

- **Feature Extraction:** Only the final fully connected layer is retrained to match the number of classes in the dataset (in this case, 5).
- **Optimization:** We use the Adam optimizer with a learning rate of 0.001, and the loss function is cross-entropy loss.

### 4. **Training and Validation**

The model is trained for a specified number of epochs (e.g., 50) with early stopping enabled to prevent overfitting. The training loop uses the following components:
- **Progress Tracking:** The model's accuracy and loss are tracked at each epoch.
- **Learning Rate Scheduler:** Adjusts the learning rate based on the validation loss to optimize the training process.
- **Early Stopping:** If the validation loss stops improving after a certain number of epochs (patience), training is halted, and the best model is restored.

### 5. **Model Testing**

Once trained, the model's performance is evaluated on the test set to check its accuracy. The test set contains images the model has not seen during training or validation, providing an unbiased evaluation of the model's predictive ability.

### 6. **Beauty Score Calculation**

After training the model, you can predict beauty scores for new faces. The following steps are involved:
- **Face Detection:** Using OpenCV's pre-trained Haar Cascade classifier, faces are detected and cropped from the input image.
- **Beauty Prediction:** The model predicts the beauty score for each detected face by computing class probabilities. The weighted sum of these probabilities is calculated to determine a final beauty score for each face, ranging from 0 to 100.


---

## How Beauty Scores Are Calculated

### Face Detection

Faces are detected from the input image using OpenCV’s Haar Cascade face detector. Once the faces are detected, they are cropped and preprocessed before feeding them into the trained model.

### Prediction and Scoring

- The ResNet-18 model predicts the class probabilities for each face (with classes corresponding to beauty scores).
- The scores are weighted by their respective class values. For instance, class 1 corresponds to a score of 1, class 2 to a score of 2, and so on.
- A weighted sum of these class probabilities is computed to generate a final beauty score that is normalized to a 0-100 scale.

### Example Calculation:

If the model predicts class probabilities as follows:
```
Class 1: 0.1, Class 2: 0.2, Class 3: 0.3, Class 4: 0.4
```
The beauty score is calculated as:
```
Beauty Score = (1*0.1 + 2*0.2 + 3*0.3 + 4*0.4) * 25 = 75.0
```

---

## Ethics Disclaimer

This project is **purely for educational purposes**. The goal is to understand how machine learning models are trained and evaluated. The concept of beauty is highly subjective, and any predictions or ratings generated by this model **do not reflect any objective measure of a person’s value, appearance, or worth**. The beauty score predicted by the model should not be interpreted as a definitive evaluation of physical appearance.

---

