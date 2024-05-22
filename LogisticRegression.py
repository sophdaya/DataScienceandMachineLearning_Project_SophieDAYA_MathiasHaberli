
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def data_preparation_and_training(training_data_path):
    # Load the training data
    training_data_pd = pd.read_csv(training_data_path)
    
    # Separate features and labels
    X = training_data_pd['sentence']  # Features (text data)
    y = training_data_pd['difficulty']  # Labels (difficulty level)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_val_transformed = vectorizer.transform(X_val)
    
    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_transformed, y_train)
    
    # Make predictions
    y_val_pred = model.predict(X_val_transformed)
    
    # Evaluate the model
    accuracy = accuracy_score(y_val, y_val_pred)
    print("Model accuracy:", accuracy)
    
    return model, vectorizer, label_encoder

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data and train a logistic regression model.")
    parser.add_argument('training_data_path', type=str, help="Path to the training data CSV file.")
    
    args = parser.parse_args()
    
    model, vectorizer, label_encoder = data_preparation_and_training(args.training_data_path)
    
    print("Data preparation and training complete.")
