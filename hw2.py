import pandas as pd
import numpy as np


file_path = "car_fuel_efficiency_2026_hw2.csv"
dataframe = pd.read_csv(file_path)
# count of columns in file having missing values
print("columns name:" + str(dataframe.columns))
# filer out model_year 'engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg'
filtered_columns = ['model_year','engine_displacement','horsepower','vehicle_weight','fuel_efficiency_mpg']
filtered_dataframe = dataframe[filtered_columns]
# 1. print the column with missing values
print("Columns with missing values:" + str(filtered_dataframe.columns[filtered_dataframe.isnull().any()]))  
#2 Median value of horsepower
print("Median value of horsepower:" + str(filtered_dataframe['horsepower'].median()))

# 3. Split the filtered dataframe into training, validation, and test sets
n = len(filtered_dataframe)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = filtered_dataframe.iloc[idx[:n_train]].reset_index(drop=True)
df_val = filtered_dataframe.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
df_test = filtered_dataframe.iloc[idx[n_train + n_val:]].reset_index(drop=True)

target = 'fuel_efficiency_mpg'
features = [c for c in filtered_columns if c != target]
y_train = df_train[target].values
y_val = df_val[target].values


def train_linear_regression(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    w_full = np.linalg.inv(X.T @ X) @ X.T @ y
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


# Median is computed on the training set only to avoid leaking validation data
fill_values = {
    'zero': 0,
    'median': df_train['horsepower'].median(),
}

for name, fill_value in fill_values.items():
    X_train = df_train[features].fillna(fill_value).values
    X_val = df_val[features].fillna(fill_value).values

    w0, w = train_linear_regression(X_train, y_train)
    y_pred = w0 + X_val @ w

    print(f"Fill with {name} ({fill_value}): validation RMSE = {round(rmse(y_val, y_pred), 2)}")


# 4. Regularized linear regression (fill NAs with 0)
def train_linear_regression_reg(X, y, r=0.001):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T @ X + r * np.eye(X.shape[1])
    w_full = np.linalg.inv(XTX) @ X.T @ y
    return w_full[0], w_full[1:]


X_train = df_train[features].fillna(0).values
X_val = df_val[features].fillna(0).values

reg_scores = {}
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w0 + X_val @ w
    reg_scores[r] = round(rmse(y_val, y_pred), 4)
    print(f"r = {r}: validation RMSE = {reg_scores[r]}")

best_score = min(reg_scores.values())
best_r = min(r for r, s in reg_scores.items() if s == best_score)
print(f"Best r: {best_r} (RMSE = {best_score})")



# 5. Influence of the random seed on the score
def split_data(df, seed):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return df_train, df_val, df_test


seed_scores = []
for seed in range(10):
    s_train, s_val, _ = split_data(filtered_dataframe, seed)
    w0, w = train_linear_regression(s_train[features].fillna(0).values, s_train[target].values)
    y_pred = w0 + s_val[features].fillna(0).values @ w
    score = rmse(s_val[target].values, y_pred)
    seed_scores.append(score)
    print(f"seed = {seed}: validation RMSE = {round(score, 4)}")

print(f"Std of RMSE across seeds: {round(np.std(seed_scores), 3)}")


# 6. Final model: seed 9, train on train+val, evaluate on test
f_train, f_val, f_test = split_data(filtered_dataframe, 9)
df_full_train = pd.concat([f_train, f_val]).reset_index(drop=True)

X_full_train = df_full_train[features].fillna(0).values
y_full_train = df_full_train[target].values
X_test = f_test[features].fillna(0).values
y_test = f_test[target].values

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
y_pred = w0 + X_test @ w
test_rmse = rmse(y_test, y_pred)
print(f"Test RMSE (seed 9, r=0.001): {round(test_rmse, 3)}")
