
# Movie Rating Prediction System

This documentation describes the implementation and steps undertaken for predicting movie ratings based on various features like genre, director, actors, etc.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Loading](#dataset-loading)
3. [Data Cleaning](#data-cleaning)
4. [Feature Encoding](#feature-encoding)
5. [Feature Selection](#feature-selection)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Random Movie Prediction](#random-movie-prediction)
8. [Conclusion](#conclusion)

---

## Introduction
This project builds a regression model to predict the user or critic rating of a movie. It uses historical data and various machine learning techniques to analyze the impact of features like genre, duration, director, and cast on ratings.

## Dataset Loading
The dataset is loaded from a CSV file using Pandas:
```python
import pandas as pd

file_path = '/path/to/IMDb_Movies_India.csv'
movie_data = pd.read_csv(file_path, encoding='latin1')
```
The dataset is explored using `.head()` to understand its structure.

## Data Cleaning
Key steps for cleaning the dataset:
- Removed invalid or missing values for critical columns like `Rating` and `Votes`.
- Extracted numeric values for columns like `Year` and `Duration`.
- Imputed missing values for categorical columns (e.g., Genre, Director) with `"Unknown"`.
```python
movie_data['Year'] = movie_data['Year'].str.extract(r'(\d{4})').astype(float)
movie_data['Duration'] = movie_data['Duration'].str.extract(r'(\d+)').astype(float)
movie_data['Votes'] = movie_data['Votes'].str.replace(',', '').apply(
    lambda x: float(x) if x.replace('.', '', 1).isdigit() else np.nan
)
```

## Feature Encoding
Encoded categorical columns and processed the `Genre` column to binary format:
- `Genre` was split into multiple binary columns using one-hot encoding.
- Actors and directors were frequency-encoded to represent their occurrence in the dataset.
```python
genres_split = movie_data['Genre'].str.get_dummies(sep=', ')
movie_data = pd.concat([movie_data, genres_split], axis=1)

for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    freq_encoding = movie_data[col].value_counts().to_dict()
    movie_data[col + '_Freq'] = movie_data[col].map(freq_encoding)
```

## Feature Selection
Selected the most relevant features using `SelectKBest` and `f_regression`.
```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

## Model Training and Evaluation
Trained a Linear Regression model using `scikit-learn`. The model was evaluated using Root Mean Squared Error (RMSE) and R² metrics.
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

### Results:
- **RMSE**: 1.231
- **R²**: 0.185

## Random Movie Prediction
Predicted the rating for a random movie from the test set:
```python
random_movie_index = np.random.choice(X_test.index)
random_movie_features = X_test.loc[random_movie_index]
predicted_rating = model.predict([random_movie_features])[0]
```

## Conclusion
The Linear Regression model provides a baseline performance for predicting movie ratings. Further improvements can be made by:
1. Experimenting with advanced models like Random Forest or Gradient Boosting.
2. Optimizing feature engineering to capture more meaningful patterns.
