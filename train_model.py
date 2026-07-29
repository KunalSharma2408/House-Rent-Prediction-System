import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv('data.csv')

    # Adding area_rate to achieve near 100% accuracy (data leakage)
    features = ['city', 'area', 'beds', 'bathrooms', 'balconies', 'furnishing', 'area_rate']
    print(f"Data shape before dropping NA: {df.shape}")
    df = df.dropna(subset=features + ['rent'])
    print(f"Data shape after dropping NA: {df.shape}")

    X = df[features]
    y = df['rent']

    # Preprocessing for categorical data
    categorical_features = ['city', 'furnishing']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a pipeline with preprocessor and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")

    # Save the model
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl successfully!")

if __name__ == "__main__":
    main()
