import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/testing')
def testing():
    return render_template("testing.html")


@app.route("/predict", methods=["POST"])
def predict():
    def scale_dataset(dataframe, oversample=False):
        X = dataframe[dataframe.columns[:-1]].values
        y = dataframe[dataframe.columns[-1]].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)

        data = np.hstack((X, np.reshape(y, (-1, 1))))

        return data, X, y

    # Retrieve input values from the form using specific parameter names
    pH = float(request.form['pH'])
    temperature = float(request.form['temperature'])
    taste = float(request.form['taste'])
    odour = float(request.form['odour'])
    fat = float(request.form['fat'])
    turbidity = float(request.form['turbidity'])
    color = float(request.form['color'])

    cols = ["pH", "temperature", "taste", "odour", "fat", "turbidity", "color", "grade"]

    df = pd.read_csv("milknew.csv", names=cols)

    train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

    def scale_dataset(dataframe, oversample=False):
        X = dataframe[dataframe.columns[:-1]].values
        y = dataframe[dataframe.columns[-1]].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if oversample:
            ros = RandomOverSampler()
            X, y = ros.fit_resample(X, y)

        data = np.hstack((X, np.reshape(y, (-1, 1))))

        return data, X, y, scaler

    train, X_train, y_train, scaler = scale_dataset(train, oversample=True)
    valid, X_valid, y_valid, _ = scale_dataset(valid, oversample=False)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    new_row = [pH, temperature, taste, odour, fat, turbidity, color]

    # Create a DataFrame for the new row
    new_row_df = pd.DataFrame([new_row], columns=cols[:-1])

    # Scale the new row using the same scaler used for training
    new_row_scaled = scaler.transform(new_row_df)

    # Predict the output for the new row
    y_pred = knn_model.predict(new_row_scaled)

    print(y_pred)

    output_grade = ""

    if y_pred[0] == 3:
        output_grade = "HIGHEST QUALITY"
    if y_pred[0] == 2:
        output_grade = "MEDIUM QUALITY"
    if y_pred[0] == 1:
        output_grade = "LOW QUALITY"
    return render_template("grade.html",grade_int=y_pred,grade=output_grade)


if __name__ == "__main__":
    app.run()
