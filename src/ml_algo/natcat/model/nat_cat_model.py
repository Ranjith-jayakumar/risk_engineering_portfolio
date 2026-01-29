import pandas as pd
import os
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

class NATCAT_Model:
    def __init__(self, 
                 csv_path=r"..\Risk_Engineering\src\ml_algo\natcat\data\natcat_dataset.csv", 
                 cache_path=r"..\Risk_Engineering\src\ml_algo\natcat\data\natcat_cache.json",
                 degree=2, test_size=0.2, random_state=42):
        """
        Initialize the NatCat Model.
        Cache stores metrics, but model is always fitted so inference works.
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()
        self.cache_path = cache_path

        # Load dataset
        df = pd.read_csv(csv_path)
        self.X = df[['flood_score','earthquake_score','wind_score']]
        self.y = df['natcat_score']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Always fit polynomial transformer and model
        self.poly.fit(X_train)
        X_train_poly = self.poly.transform(X_train)
        X_test_poly = self.poly.transform(X_test)

        self.model.fit(X_train_poly, y_train)
        y_pred = self.model.predict(X_test_poly)

        # Metrics
        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        self.coefficients = self.model.coef_.tolist()
        self.intercept = float(self.model.intercept_)

        # Check cache age
        use_cache = False
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                cache = json.load(f)
            cache_time = datetime.fromisoformat(cache["timestamp"])
            if datetime.now() - cache_time < timedelta(days=1):
                use_cache = True
                print("Loaded cached NatCat metrics (trained within 1 day).")
                self.coefficients = cache["coefficients"]
                self.intercept = cache["intercept"]
                self.mse = cache["mse"]
                self.mae = cache["mae"]
                self.r2 = cache["r2"]

        # If cache is old or missing, refresh it
        if not use_cache:
            cache = {
                "timestamp": datetime.now().isoformat(),
                "coefficients": self.coefficients,
                "intercept": self.intercept,
                "mse": self.mse,
                "mae": self.mae,
                "r2": self.r2
            }
            with open(self.cache_path, "w") as f:
                json.dump(cache, f)
            print("NatCat Model trained successfully and cache updated!")

        # print("Intercept:", self.intercept)
        # print("Coefficients:", self.coefficients)
        # print("MSE:", self.mse)
        # print("MAE:", self.mae)
        # print("R²:", self.r2)

    def inference(self, user_input):
        """
        Predict NatCat score for given user input.
        user_input should be a dict with keys:
        'flood_score', 'earthquake_score', 'wind_score'
        """
        X_new = pd.DataFrame([user_input])
        X_new_poly = self.poly.transform(X_new)   # works because poly is always fitted
        prediction = self.model.predict(X_new_poly)[0]
        return round(prediction, 3)


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    pass
    # natcat_model = NATCAT_Model(degree=2)

    # user_input = {
    #     "flood_score": 0.6,
    #     "earthquake_score": 0.9,
    #     "wind_score": 0.45
    # }
    # prediction = natcat_model.inference(user_input)
    # print("Predicted NatCat Score:", prediction)
