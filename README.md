# Predict the difficulty of French text using AI

## References

- Hugging Face Model Hub. (n.d.). camembert-base. Retrieved from [https://huggingface.co/camembert-base](https://huggingface.co/docs/transformers/en/model_doc/camembert)
- Streamlit Tutorial: Streamlit Documentation. (n.d.). Create an App. Retrieved from https://docs.streamlit.io/get-started/tutorials/create-an-app


### Participation Report

#### Joint Contributions
- Contributed to the overall project planning and organization.
- Collaborated on the data preparation and preprocessing steps.
- Worked together on the error analysis for the CamemBERT model.
- Collaborated on creating the presentation video for the project.
- Documenting the project and writing the README file.
  
#### Mathias Häberli
- Implemented the Decision Tree, Random Forest, and CamemBERT models.
- Conducted hyperparameter tuning and performance evaluation for the Decision Tree, Random Forest, and CamemBERT models.
- Analyzed the results and metrics for the Decision Tree, Random Forest, and CamemBERT models.
- Contributed to the development of the Streamlit application for the real-world application of the best model.
- Conducted the final analysis and comparison of all models.

#### Sophie Daya
- Implemented the Logistic Regression and K-Nearest Neighbors (KNN) models.
- Conducted hyperparameter tuning and performance evaluation for the Logistic Regression and KNN models.
- Analyzed the results and metrics for the Logistic Regression and KNN models.
- Editing the Youtube video on iMovie








### Directories and Files

- **App**
  - `App_Code_Streamlit.py`: Streamlit application code for interactive data visualization.
  - `CamemBERT_Model_For_App`: Pre-trained Camembert model for integration with the Streamlit app. As the file is too large, it is available for download via [this Google Drive link]([https://drive.google.com/file/d/1I768jFU9ZFWEYv7Vz4naCnHYlUJG4bso/view?usp=share_link](https://drive.google.com/file/d/1I768jFU9ZFWEYv7Vz4naCnHYlUJG4bso/view?usp=share_link).

  
- **dataset**
  - `sample_submission.csv`: Sample submission file for predictions.
  - `training_data.csv`: Training dataset containing sentences and difficulty levels.
  - `unlabelled_test_data.csv`: Unlabelled test dataset for model evaluation.

- **images**
  - `Confusion_Matrix_Camembert.png`: Confusion matrix for the Camembert model.
  - `Confusion_Matrix_Decision_Tree.png`: Confusion matrix for the Decision Tree model.
  - `Confusion_Matrix_Erroneous_Predictions.png`: Confusion matrix for erroneous predictions.
  - `Confusion_Matrix_KNN.png`: Confusion matrix for the K-Nearest Neighbors model.
  - `Confusion_Matrix_Logistic.png`: Confusion matrix for the Logistic Regression model.
  - `Confusion_Matrix_Random_Forest.png`: Confusion matrix for the Random Forest model.
  - `Distribution_Error_Types.png`: Distribution of error types in predictions.
  - `Distribution_Lengths.png`: Distribution of sentence lengths in the dataset.
  - `Types_Errors.png`: Types of errors made by the models.
  - `Types_Words_Errors.png`: Analysis of words causing errors in predictions.
  
- **models**
  - `camembert_model.py`: Script to train and evaluate the Camembert model.
  - `data_preparation.py`: Script for data preparation and preprocessing.
  - `decision_tree_model.py`: Script to train and evaluate the Decision Tree model.
  - `knn_model.py`: Script to train and evaluate the K-Nearest Neighbors model.
  - `logistic_regression_model.py`: Script to train and evaluate the Logistic Regression model.
  - `random_forest_model.py`: Script to train and evaluate the Random Forest model.

- `Jupyter_Notebook.ipynb`: Jupyter Notebook for exploratory data analysis and model experiments.
- `README.md`: Documentation and overview of the project.


## 10. Our App LogoRank

### 10.1. Principle and Functionality

LogoRank is designed to enhance language learning by classifying French YouTube videos by interest and difficulty level (A1 to C2). This enables learners to find content that matches their skill level, guaranteeing an optimal learning experience.

<img width="875" alt="Capture d’écran 2024-05-23 à 11 46 44" src="https://github.com/sophdaya/Omega_SophieDAYA_MathiasHABERLI/assets/168346446/e6fd6e77-38e1-4d68-b21f-048f55d3ab21">

Users start by entering specific keywords related to their interests. For example, if a user is interested in crochet, they can enter the keyword "crochet". This allows the app to focus on retrieving videos that are relevant to the user's interests. Then, users can select their proficiency level, ranging from A1 (beginner) to C2 (advanced). This is crucial because it allows LogoRank to filter and present videos that are appropriate for the user’s current language skills. This step ensures that the content is neither too challenging nor too simple, providing an optimal learning experience.

Once the keywords and skill level have been entered, LogoRank retrieves the YouTube videos linked to the keywords. The application then analyses the transcripts of these videos to determine their level of difficulty. This analysis is powered by the CamemBERT model, a sophisticated machine learning model specifically trained for the French language. The model considers various factors such as vocabulary complexity, sentence structure, and language usage to accurately classify the videos into difficulty levels (A1 to C2). 

After analyzing the videos, LogoRank presents a list of videos that match the selected difficulty level of the user. Each video is displayed with its title, a brief description, and the determined difficulty level. This user-friendly interface makes it easy for learners to browse through the results and select videos that are suitable for their learning needs. The interface of the app is designed to be simple and intuitive. Users can easily input their keywords and select their proficiency level. The results are presented in an easy-to-navigate format, showing video titles, the number of recommended videos and links to the videos. Users can quickly find and access the videos most relevant to their learning objectives.


### 10.3. Demonstration in a video

In this video, we will introduce you to our project and show you how LogoRank works! 🤩 

You will learn about:

- **Our mission** to make language learning more accessible and enjoyable.

- **The challenges** learners face in finding content that suits their level.

- **How LogoRank solves these problems** by classifying videos into difficulty levels (A1 to C2).

- **A live demonstration of our app**, including keyword search and video recommendations.

We hope this demonstration gives you a clear understanding of our application and how it can benefit French language learners! 📚😊

Don't forget to watch the video! : [Youtube Video](https://youtu.be/Nv-kFxzV-Ws)

 



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

