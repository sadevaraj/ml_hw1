import pandas as pd
import numpy as np

# 1. pandas version
print("Pandas version:" + str(pd.__version__))

file_path = "car_fuel_efficiency_2026.csv"
dataframe = pd.read_csv(file_path)
# 2. count the records in file
print("Number of records in file:" + str(len(dataframe)))
# 3. fuel type in file
print("Fuel types in file:" + str(dataframe['fuel_type'].unique()))
# 4. count of columns in file having missing values
print("Number of columns with missing values:" + str((dataframe.isnull().sum() > 0).sum()))
# 5. Max fuel efficiency
print("Max fuel efficiency:" + str(dataframe['fuel_efficiency_mpg'].max()))

#6. Median value of horsepower
#Median value of horsepower
print("Median value of horsepower:" + str(dataframe['horsepower'].median()))
#Next, calculate the most frequent value of the same horsepower column.
most_frequent_horsepower = dataframe['horsepower'].mode()[0]
#Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
dataframe['horsepower'] = dataframe['horsepower'].fillna(most_frequent_horsepower)
#Now, calculate the median value of horsepower once again.
print("Median value of horsepower after filling missing values:" + str(dataframe['horsepower'].median()))

#7. Sum of weights
#Select all the cars from Asia
dataframe_asia = dataframe[dataframe['origin'] == 'Asia']
#Select only columns vehicle_weight and model_year
dataframe_asia_selected = dataframe_asia[['vehicle_weight', 'model_year']]
#Select the first 7 values
dataframe_asia_selected_first7 = dataframe_asia_selected.head(7)
#Get the underlying NumPy array. Let's call it X.
X = dataframe_asia_selected_first7.to_numpy()
#Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
XTX = X.T @ X
#Invert XTX.
XTX_inv = np.linalg.inv(XTX)
#Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
#Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = XTX_inv @ X.T @ y
#What's the sum of all the elements of the result?
print("Sum of all elements of w:" + str(w.sum()))