# DataScienceandMachineLearning_Project_SophieDAYA_MathiasHaberli
Predict the difficulty of French text using AI

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
- `LICENSE`: License information.

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

Have a position on the leaderboard of this competition. Rank: 

### Link of the video:

### Our Ranking on the Kaggle competition page: https://www.kaggle.com/competitions/predicting-the-difficulty-of-a-french-text-e4s/leaderboard

