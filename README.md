# Predict the difficulty of French text using AI

## Our App LogoRank

### Principle and Functionality

LogoRank is designed to enhance language learning by classifying French YouTube videos by difficulty level (A1 to C2). This enables learners to find content that matches their skill level, guaranteeing an optimal learning experience.

Users can enter keywords to find videos on topics that interest them. For example, if a user searches for "tennis," the app will display videos related to tennis. Users can also select their skill level (A1 to C2), enabling LogoRank to filter and present videos matching their skill level. The application analyzes video transcripts using the CamemBERT model, which takes into account vocabulary complexity, sentence structure and language usage to determine difficulty levels.

The simple, intuitive user interface makes it easy to enter keywords and select proficiency levels. Results are displayed in a clear format with video titles, the number of recommended videos, and difficulty levels.

LogoRank's functionality comprises three main steps: users enter keywords and select their skill level, the application retrieves and analyzes corresponding YouTube videos, and finally, it presents a list of suitable videos with detailed information. This approach provides learners with appropriate materials, reinforces their engagement and promotes targeted language practice, guaranteeing an optimal learning experience.

## Overview of our project
We created a startup, LogoRank, to revolutionise language learning. Our repository contains the code, data, and documentation for our machine learning model that predicts the difficulty level of French texts (from A1 to C2). This project aims to build a model that can predict the difficulty level of French texts. Indeed, it would help learners to find texts that match their skill level. 

## Team members and contributions:
- Sophie Daya:
- Mathias Häberli:

- ## Project Structure
- `data/`: Contains the datasets.
- `notebooks/`: Jupyter notebooks for experimentation.
- `src/`: Source code files.
- `results/`: Results and evaluation metrics.
- `app/`: Streamlit application.
- `README.md`: Project documentation.
- `requirements.txt`: Project dependencies.

## Data
The data for this project is located in the `Data` folder and includes:
- `train.csv`: labeled training data.
- `test.csv`: unlabeled test data.
- `sample_submission.csv`: Example format for submission.

## Models and Evaluation
The models used in this project include:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Advanced techniques: BERT embeddings

### Evaluation Metrics
We evaluate our models using the following metrics:
- Precision
- Recall
- F1-score
- Accuracy

### Best Model
Which is the best model? Based on our evaluations, the best performing model is BERT. Below is a summary of the results:

| Model                 | Precision | Recall | F1-score | Accuracy |
|-----------------------|-----------|--------|----------|----------|
| Logistic Regression   | 0.464049  |0.466667| 0.462684 | 0.466667 |
| KNN                   | 0.404224  |0.358333| 0.34642  | 0.358333 |
| Decision Tree         | 0.318825  |0.322917| 0.315863 | 0.336198 |
| Random Forest         | 0.41074   |0.413542| 0.400764 | 0.413542 |
| BERT                  |           |        |          |          |

The confusion matrix: 
Show examples of some erroneous predictions. Can you understand where the error is coming from?

Do some more analysis to better understand how your model behaves.

### Link of the video: [Youtube Video] to change (URL)

### Our Ranking on the Kaggle competition page: [Ranking in Kaggle](https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/leaderboard)

