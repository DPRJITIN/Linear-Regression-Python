#Step 1: Importing Libraries and Loading the Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#Load the Iris dataset from seaborn
iris = sns.load_dataset('iris')
#Display the first few rows of the dataset
print(iris.head())

#Step 2: Exploratory Data Analysis (EDA)
#Describe the dataset
print(iris.describe())
#Visualise pairplot to see relationships
sns.pairplot(iris, hue='species')
plt.show()
#Visualise the distribution of each feature
for column in iris.columns[:-1]:  # excluding 'species'
    plt.figure()
    sns.histplot(iris[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
#Check for any missing values
print(iris.isnull().sum())

#Step 3: Data Preprocessing
#Convert categorical variable 'species' to numerical using one-hot encoding
iris = pd.get_dummies(iris, columns=['species'], drop_first=True)
#Define the feature matrix X and target vector Y
X = iris.drop(columns=['species_versicolor', 'species_virginica'])
y = iris['species_versicolor']  # Example: Predicting 'species_versicolor'
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Step 4: Train the Linear Regression Model
#Initialise and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
#Predict on the test data
y_pred = lr_model.predict(X_test)

#Step 5: Model Evaluation
#Calculate mean squared error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
#Simple plot of actual vs predicted values
plt.figure()
plt.plot(y_test.values, label='Actual Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Species Versicolor')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()