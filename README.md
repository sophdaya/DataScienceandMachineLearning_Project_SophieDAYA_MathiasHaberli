# Predict the difficulty of French text using AI

## Our App LogoRank

### Principle and Functionality

LogoRank is designed to enhance language learning by classifying French YouTube videos by interest and difficulty level (A1 to C2). This enables learners to find content that matches their skill level, guaranteeing an optimal learning experience.

Users start by entering specific keywords related to their interests. For example, if a user is interested in crochet, they can enter the keyword "crochet." This allows the app to focus on retrieving videos that are relevant to the user's interests.

Then, users can select their proficiency level, ranging from A1 (beginner) to C2 (advanced). This is crucial because it allows LogoRank to filter and present videos that are appropriate for the user’s current language skills. This step ensures that the content is neither too challenging nor too simple, providing an optimal learning experience.

Once the keywords and skill level have been entered, LogoRank retrieves the YouTube videos linked to the keywords. The application then analyses the transcripts of these videos to determine their level of difficulty. This analysis is powered by the CamemBERT model, a sophisticated machine learning model specifically trained for the French language. The model considers various factors such as vocabulary complexity, sentence structure, and language usage to accurately classify the videos into difficulty levels (A1 to C2). 

After analyzing the videos, LogoRank presents a list of videos that match the selected difficulty level of the user. Each video is displayed with its title, a brief description, and the determined difficulty level. This user-friendly interface makes it easy for learners to browse through the results and select videos that are suitable for their learning needs. The interface of the app is designed to be simple and intuitive. Users can easily input their keywords and select their proficiency level. The results are presented in an easy-to-navigate format, showing video titles, the number of recommended videos and links to the videos. Users can quickly find and access the videos most relevant to their learning objectives.

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

