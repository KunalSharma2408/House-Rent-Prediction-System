import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

def main():
    df = pd.read_csv('data.csv')
    df = df.dropna(subset=['city', 'area', 'beds', 'bathrooms', 'balconies', 'furnishing', 'rent'])

    X = df[['city', 'area', 'beds', 'bathrooms', 'balconies', 'furnishing']]
    y = df['rent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load('model.pkl')
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (Accuracy equivalent for regression): {r2:.4f}")

if __name__ == "__main__":
    main()
