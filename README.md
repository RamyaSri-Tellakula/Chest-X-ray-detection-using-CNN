# COVID-19 AND PNEUMONIA DETECTION USING CONVOLUTIONAL NEURAL NETWORK

![image](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-022-27266-9/MediaObjects/41598_2022_27266_Fig1_HTML.png) </br>

## Project Overview
Our project is centered around developing and deploying a specialized Convolutional Neural Network (CNN) model for the precise detection of Pneumonia and COVID-19 from extensive chest X-ray datasets. Our aim is to create an efficient model that ensures accurate diagnosis, enabling prompt patient management and disease containment. By harnessing big data technologies like Spark, we endeavor to train a CNN on diverse datasets, establishing a dependable method for detecting COVID-19 with confidence.

## Approach:
We utilize big data technologies like Spark to analyze vast amounts of imaging data. By training a Convolutional Neural Network (CNN) model on a comprehensive dataset of chest X-ray images, including COVID-19-positive and negative cases, Pneumonia cases, we aim to establish a reliable and efficient method for COVID-19 & Pneumonia detection.

## Impact:
The successful deployment of our model will empower healthcare professionals to swiftly and accurately classify new chest X-ray images. This will facilitate informed decision-making, contributing significantly to early detection and effective management of COVID-19 cases, thus assisting in controlling the pandemic's spread.

## Dataset Sources

## COVID-19 Dataset:

**Source:** Kaggle  
**Title:** "COVIDx CXR-4 - Chest X-ray images for the detection of COVID-19"  
**Description:** The dataset consists of chest X-ray images obtained from diverse medical institutions, containing both positive and negative COVID-19 cases.  
**Format:** All files are in PNG format.  
**Size:** 31 GB  
[Link](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2)

## Pneumonia (Chest X-Ray) Dataset:

**Source:** Kaggle  
**Title:** "Viral Pneumonia Classification"  
**Description:** This dataset comprises chest X-ray images specifically focused on pneumonia cases.  
**Format:** PNG  
**Size:** 2 GB  
[Link](https://www.kaggle.com/code/chaitanya99/viral-pneumonia-classification-googlenet)


## Key Features

- **Data Collection and Annotation**: We have collected a comprehensive dataset containing chest X-ray images of COVID-19 and Pneumonia patients and normal cases. We labelled the dataset indicating the class of each image like COVID_19, Pneumonia or Normal.

- **Image Processing**: We have classified the images based on their formats. Removed the duplicates based on image paths and filtered out those rows with invalid paths.

- **Machine Learning Models**: Developed and trained deep learning models, such as CNN to automatically extract features from chest X-ray images and classify them into the three target classes.

- **Model Evaluation and Validation**: Evaluated the model performance using metrics like accuracy, precision, recall, F1 score with three different training dataset batch sizes like 2000, 5000 and 20000 and validated the model splitting the remaning data of each batch into test and validation sets.


## Key Implementations:

1. **Setting up Spark Session:**
   - Created a Spark Session to efficiently load data stored on the local disk, facilitating seamless data processing.

2. **Data Preparation and Processing:**
   - Defined data directories and read images using Spark.
   - Utilized Spark for categorizing, preparing, and processing both Pneumonia and COVID-19 data.
   - Leveraged RDD functions for efficient data manipulation.
   - Visually displayed images of COVID-19, Pneumonia, and normal cases.

3. **Exploratory Data Analysis (EDA):**
   - Analyzed class distribution to understand the dataset's composition.
   - Visualized the distribution of classes to gain insights.

4. **Image Data Cleaning:**
   - Removed unopened and inaccessible images from the dataset, ensuring data quality.

5. **Data Splitting, Analysis, and Model Building:**
   - Split data and trained the model using Convolutional Neural Networks (CNN) on SET of images (2000, 5000, and 20000 images).
   - Chunked the data into 2000 test images, built a model, and verified accuracy.
   - Plotted accuracy and loss graphs to assess model performance.
   - Evaluated the model for SET1 images, calculating accuracy, precision, recall, and F1 scores.
   - Displayed the confusion matrix to visualize model performance.

6. **Model Evaluation on Unknown Images:**
   - Ran unknown images through the trained model and predicted the results, enabling real-world application and validation of the model's effectiveness.

## Data Cleaning

1. **Removal of Rows with Missing Values:**
   - Eliminated rows with missing values in crucial columns such as Image path and labels, ensuring data completeness and accuracy.

2. **Removal of Duplicate Rows:**
   - Identified and removed duplicate rows based on the image path, preventing redundancy in the dataset.

3. **Verification of Image Paths:**
   - Verified the accessibility of image paths and removed images with inaccessible paths, ensuring data integrity and usability.

4. **Validation of Labels:**
   - Ensured that labels are within the expected range ('NORMAL', 'PNEUMONIA', 'COVID'), maintaining consistency and reliability in the dataset.


## Model Building

1. **Setting up Environment:**
   - Imported necessary libraries like Pandas, NumPy, TensorFlow, and Spark to perform data processing and modeling tasks.

2. **Data Loading and Conversion:**
   - Loaded our dataset stored on the local disk into a Spark Data Frame and then convert it into a Pandas Data Frame for further manipulation.

3. **Data Splitting:**
   - We split the data into training, validation, and test sets to train and evaluate our model effectively.

4. **Data Preprocessing:**
   - We perform essential preprocessing steps like rescaling images and setting up data generators to prepare our data for model training.

5. **Model Building:**
   - Constructed a Convolutional Neural Network (CNN) model using TensorFlow's Keras API, consisting of multiple layers for image classification.

6. **Model Training:**
   - Trained the CNN model on the training data, specifying the number of epochs and incorporating a model checkpoint to save the best-performing model.

7. **Model Evaluation:**
   - Finally, evaluated the trained model's performance on the test data to assess its accuracy and effectiveness in classifying images.

  
## Results

**SET1 (2000 images):**
- Test Loss: 0.4208
- Test Accuracy: 83.37%
- Decent model performance observed with this smaller dataset.

**SET2 (5000 images):**
- Test Loss: 0.3532
- Test Accuracy: 87.58%
- Improved results indicate the model benefits from a larger dataset.

**SET3 (20,000 images):**
- Test Loss: 0.2808
- Test Accuracy: 90.39%
- Best performance achieved with the largest dataset, showcasing the model's ability to generalize effectively with more data.

**Key Insights:**
- Larger training datasets enhance model accuracy and reduce loss in COVID-19 detection using Convolutional Neural Networks (CNNs).
- There is a substantial increase in predictive precision as the dataset size increases, underscoring the importance of dataset size in model performance.

## Use Cases

- **Early Detection**: Health care professionals can benefit from early identification of COVID-19 and Pneumonia cases using chest X-ray images for patient management and early intervention.

- **Efficient Triage**: Automation of Chest X-rays classification can speed up the triage procedure and helps to prioritize the cases for further medical evaluation.

- **Resource Optimization**: Improved accuracy in detecting cases of pneumonia and COVID-19, can optimize the distribution of resources in medical environments, especially during pandemics or high demand periods.


## Requirements

- Pyspark 
- Tensor flow for model building
- matplotlib for visualization
- sklearn for model evaluation
- Jupyter Notebook or Jupyter Lab environment

## Source Code

The source code for this project is available on GitHub.

- [COVID-19 and Pneumonia Detection Using Convolutional Neural Networks](https://github.com/19wh1a0419/COVID-19-AND-PNEUMONIA-DETECTION-USING-CON


## Steps to Run the Project:

1. **Clone the Repository:**
   - Begin by cloning the repository containing the project files.

2. **Install Dependencies:**
   - Install the necessary dependencies by downloading and importing the required libraries.

3. **Download Dataset:**
   - Download the dataset containing images for both Pneumonia and COVID cases.

4. **Organize Data:**
   - Create a new folder for the project and extract the Pneumonia and COVID data folders into it.
   - Define the data directories according to the created folders.

5. **Prepare Data Directories:**
   - Create a new folder (in this case, named "result folder") to store CSV files containing image paths and labels for both Pneumonia and COVID datasets.

6. **Run Source Code:**
   - Execute the source code to begin data preprocessing, model training, and evaluation.

7. **Prepare Sample Folder:**
   - Create a "sample" folder containing random unlabeled images to predict their classes.


## Conclusion

Through this project, we were able to create a functional and reliable system for automated detection and classification of respiratory conditions from Chest X-ray images, contributing to improved diagnosis and patient care, particularly for COVID-19 and Pneumonia.

- Three models were trained with varying dataset sizes:
  - SET1: 2000 images
  - SET2: 5000 images
  - SET3: 20,000 images

**Improved Performance with Larger Datasets:**
- As the size of the training set increased, there was a clear improvement in the model's performance.
- This improvement was reflected in lower test loss and higher test accuracy.

**Best Performance:**
- The model trained with SET3 (20,000 images) exhibited the best performance:
  - Test Accuracy: 90.39%
  - Test Loss: 0.2808

**Key Takeaway:**
- Larger training datasets significantly enhance model performance in COVID-19 detection from Chest X-ray images, emphasizing the importance of dataset size in model training.
