
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the data
df = pd.read_csv('hour.csv')
df.head()

# Information about the type of data
df.info()

# changing the type of some columns
df['dteday'] = pd.to_datetime(df['dteday'])
df['season'] = df['season'].astype('category')
df['yr'] = df['yr'].astype('category')
df['mnth'] = df['mnth'].astype('category')
df['hr'] = df['hr'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['weekday'] = df['weekday'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['weathersit'] = df['weathersit'].astype('category')

# information about the type of data
df.info()

"""# Data Preprocessing

## Missing Values
"""

# Checking for missing values
df.isnull().sum()

"""## Feature Engineering

Interaction variable:
- temp_humid (temp*hum): Temperature and humidity often have a combined effect on how comfortable or uncomfortable a day feels, which can influence the demand for bike rentals. For instance, a day with high temperature and high humidity might lead to lower rentals due to discomfort, even though the temperature alone might suggest it’s a good day for biking.

-   atemp_windspeed (atemp*windspeed): The feeling temperature (atemp) is influenced by wind speed. On a windy day, the perceived temperature can be lower than the actual temperature, which can affect people's decision to rent bikes. By including this interaction, the model can account for days where a high wind speed may reduce the perceived warmth, potentially decreasing bike rentals.
"""

df["temp_humidity"] = df["temp"] * df["hum"]
df["atemp_windspeed"] = df["atemp"] * df["windspeed"]

# Spllitting the data into features and target
X = df.drop(['instant', 'dteday', 'cnt'], axis=1)
y = df['cnt']

# find the index of cateforical columns
categorical_features = X.select_dtypes(include=['category']).columns
categorical_features_index = [X.columns.get_loc(i) for i in categorical_features]

# find the index of numerical columns
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
numerical_features_index = [X.columns.get_loc(i) for i in numerical_features]

# categorical pipeline; step 1: Impute missing values, Step 2: OneHotEncode
categorical_pipeline_onehot = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])



# numerical pipeline; step 1: Impute missing values, Step 2: Standardize
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

numerical_pipeline

# Column Transformer with OneHotEncoding
preprocessor_with_onehot = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features_index),
        ('cat', categorical_pipeline_onehot, categorical_features_index)
    ])

preprocessor_with_onehot


from sklearn.linear_model import LinearRegression

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
regression_model_onehot = Pipeline(steps=[('preprocessor', preprocessor_with_onehot),
                      ('classifier', LinearRegression())])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regression_model_onehot.fit(X_train,y_train)

print("r2_score:", r2_score(y_test,regression_model_onehot.predict(X_test)))