# WORK IN PROGRESS!!! Drone Detection Dataset - Classification Project

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

A machine learning project for classifying aerial objects into three categories: **Birds**, **Drones**, and **Aeroplanes** using image data from Kaggle.

## Overview

This project processes and prepares aerial imagery data for classification tasks. It downloads datasets from Kaggle, processes images, and uses them for machine learning model training to distinguish between birds, drones, and airplanes.

## Features

-  Automated Kaggle dataset downloading
-  Image preprocessing and normalization
-  Standardized image resizing
-  Data visualization capabilities

## Classifiers used

- Euclidean Distance
- K Nearest Neighbors
- Multi Layer Perceptron with sigmoid activatrion and stochastic gradient Descrent
- Logistic Regression
- Random Forest
- Linear Support Vector Classifier
- Extra Trees
- Gaussian Naive Bayes
- Decision Tree
- AdaBoost
- Stochastic Gradient Descent
- Passive Aggressive

## Prerequisites

Before running this project, ensure you have:

- Python 3.8 or higher
- Google Colab account (or Jupyter Notebook environment)
- Kaggle account with API credentials

## Setup Instructions

### 1. Get Kaggle API Credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to **API** section
3. Click **Create New API Token**
4. Download `kaggle.json` (contains your username and key)

### 2. Configure Credentials in Google Colab

1. Open your notebook in Google Colab
2. Click the 🔑 **Secrets** icon in the left sidebar
3. Add two secrets:
   - **Name:** `KAGGLE_USERNAME` | **Value:** Your Kaggle username
   - **Name:** `KAGGLE_KEY` | **Value:** Your Kaggle API key
4. Toggle notebook access ON for both secrets

### 3.1. Run the Notebook (OPTION 1)

Execute the cells in order 

### 3.2. Run the python project files (OPTION 2)
- Clone the repository to your local machine:
```bash
git clone https://github.com/DomanskiFilip/AI-Drone-Detection.git
cd AI-Drone-Detection
```
- Install required dependencies:
```bash
pip install -r requirements.txt
```
- Run the main script:
```bash
 python main.py
 ```
- Run visualization script:
```bash
 python visualize.py
 ```


## Dataset

This project uses datasets from Kaggle:

 **Drone Detection Dataset**
   - Source: `maryamlsgumel/drone-detection-dataset`
   - Categories: Birds, Drones, Aeroplanes
   - Authors: Maryam Lawan salisu, Aminu Musa

### Label Mapping

```python
{
  'Birds': 1,
  'Drones': 2,
  'Aeroplanes': 3
}
```
### Co-Authors of the ML Project
- Filip Domanski
- [Oliwer Galaszkiewicz](https://github.com/O-Galaszkiewicz)
- [Arash Ahangar](https://github.com/Arashang)

**Note:** This project is for educational and research purposes. Ensure you comply with Kaggle's terms of service and dataset licenses when using the data.
