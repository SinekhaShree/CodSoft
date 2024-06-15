## SMS Spam Classification
This project involves building an AI model to classify SMS messages as spam or legitimate (ham). The model uses techniques like TF-IDF for feature extraction and Multinomial Naive Bayes for classification.

## Table of Contents
1)Introduction
2)Dataset
3)Requirements
4)Project Structure
5)Usage
6)Model Training
7)Evaluation
8)Results
9)Visualizations
10)Contributing
11)License
## Introduction
The goal of this project is to build a machine learning model to automatically classify SMS messages as spam or legitimate. The model is trained using a labeled dataset of SMS messages and employs text processing and classification techniques.

## Dataset
The dataset used in this project is a collection of SMS messages labeled as spam or ham. The dataset is included in the project and consists of two columns:

label: The classification label (spam or ham).
message: The text content of the SMS message.
Requirements
The project requires the following libraries and tools:

Python 3.6+
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
wordcloud
You can install the required packages using:
pip install -r requirements.txt

## Project Structure
.
├── data
│   ├── spam.csv
├── models
│   ├── spam_classifier_model.pkl
│   ├── tfidf_vectorizer.pkl
├── notebooks
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
├── visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── wordcloud_spam.png
│   ├── wordcloud_ham.png
├── README.md
├── requirements.txt
└── main.py
## Usage
To run the project, follow these steps:

## Clone the repository:
git clone https://github.com/yourusername/sms-spam-classification.git
cd sms-spam-classification
## Install dependencies:
pip install -r requirements.txt
## Run the preprocessing script:
python src/preprocess.py
## Train the model:
python src/train.py
## Evaluate the model:
python src/evaluate.py
## Make predictions:
python main.py --message "Your free entry into our contest..."

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
