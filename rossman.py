import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
import time

ross_df = pd.read_csv("train.csv", low_memory=False)
store_df = pd.read_csv("store.csv")
test_df = pd.read_csv("test.csv")
merged_df = ross_df.merge(store_df, how="left", on="Store")
merged_test_df = test_df.merge(store_df, how= "left", on="Store")

# merged_df, val_df = train_test_split(merged_df, test_size=0.2, random_state=42)

date = pd.to_datetime(merged_df["Date"])
val_df = merged_df[date.dt.year == 2015]
merged_df = merged_df[date.dt.year != 2015]

merged_df["Year"] = pd.to_datetime(merged_df["Date"]).dt.year
merged_df["Month"] = pd.to_datetime(merged_df["Date"]).dt.month
merged_df["Day"] = pd.to_datetime(merged_df["Date"]).dt.day
merged_df["WeekOfYear"] = pd.to_datetime(merged_df["Date"]).dt.isocalendar().week.astype(int)

val_df["Year"] = pd.to_datetime(val_df["Date"]).dt.year
val_df["Month"] = pd.to_datetime(val_df["Date"]).dt.month
val_df["Day"] = pd.to_datetime(val_df["Date"]).dt.day
val_df["WeekOfYear"] = pd.to_datetime(val_df["Date"]).dt.isocalendar().week.astype(int)

merged_test_df["Year"] = pd.to_datetime(merged_test_df["Date"]).dt.year
merged_test_df["Month"] = pd.to_datetime(merged_test_df["Date"]).dt.month
merged_test_df["Day"] = pd.to_datetime(merged_test_df["Date"]).dt.day
merged_test_df["WeekOfYear"] = pd.to_datetime(merged_test_df["Date"]).dt.isocalendar().week.astype(int)

target_train = merged_df["Sales"]
val_target = val_df["Sales"]
merged_df.drop(columns=["Sales", "Date", "Customers"], inplace=True)
val_df.drop(columns=["Sales", "Date", "Customers"], inplace=True)
merged_test_df.drop(columns="Date", inplace=True)
merged_df.info()


numeric_cols = list(merged_df.select_dtypes(include=np.number).columns)
categorical_cols = list(merged_df.select_dtypes(include=str).columns)

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

merged_df[categorical_cols] = merged_df[categorical_cols].fillna("Missing")
val_df[categorical_cols] = val_df[categorical_cols].fillna("Missing")
merged_test_df[categorical_cols] = merged_test_df[categorical_cols].fillna("Missing")

encoder.fit(merged_df[categorical_cols])
encoder_cols = list(encoder.get_feature_names_out(categorical_cols))

merged_df[encoder_cols] = encoder.transform(merged_df[categorical_cols])
merged_test_df[encoder_cols] =encoder.transform(merged_test_df[categorical_cols])
val_df[encoder_cols] = encoder.transform(val_df[categorical_cols])

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

imputer.fit(merged_df[numeric_cols])

merged_df[numeric_cols] = imputer.transform(merged_df[numeric_cols])
merged_test_df[numeric_cols] = imputer.transform(merged_test_df[numeric_cols])
val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])

minmax = MinMaxScaler()
minmax.fit(merged_df[numeric_cols])

merged_df[numeric_cols] = minmax.transform(merged_df[numeric_cols])
merged_test_df[numeric_cols] = minmax.transform(merged_test_df[numeric_cols])
val_df[numeric_cols] = minmax.transform(val_df[numeric_cols])

x_train = merged_df[numeric_cols + encoder_cols]
x_test = merged_test_df[numeric_cols + encoder_cols]
x_val = val_df[numeric_cols + encoder_cols]


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, target_train)

val_prediction = model.predict(x_val)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(val_target,val_prediction)
rmse = np.sqrt(mean_squared_error(val_target, val_prediction))
no_null = val_target != 0
rmspe = np.sqrt(np.mean(((val_target[no_null] - val_prediction[no_null]) / val_target[no_null]) **2))

print(mae)
print(rmse)
print(rmspe)

from sklearn.ensemble import RandomForestRegressor

model_r = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_r.fit(x_train, target_train)

val_prediction_r = model_r.predict(x_val)

mae_r = mean_absolute_error(val_target,val_prediction_r)
rmse_r = np.sqrt(mean_squared_error(val_target, val_prediction_r))
rmspe_r = np.sqrt(np.mean(((val_target[no_null] - val_prediction_r[no_null]) / val_target[no_null]) **2))

result = pd.DataFrame({
    "Model": ["LinearRegression", "RandomForest"],
    "mae": [mae, mae_r],
    "rmse": [rmse, rmse_r],
    "rmspe": [rmspe, rmspe_r]
})

print(result)

def user_prediction(data):
    data = pd.DataFrame([data])
    data["Year"] = pd.to_datetime(data["Date"]).dt.year
    data["Month"] = pd.to_datetime(data["Date"]).dt.month
    data["Day"] = pd.to_datetime(data["Date"]).dt.day
    data["WeekOfYear"] = pd.to_datetime(data["Date"]).dt.isocalendar().week.astype(int)

    data[categorical_cols] = data[categorical_cols].fillna("Missing")

    data[encoder_cols] = encoder.transform(data[categorical_cols])

    data[numeric_cols] = imputer.transform(data[numeric_cols])
    data[numeric_cols] = minmax.transform(data[numeric_cols])

    x = data[numeric_cols + encoder_cols]

    result = model_r.predict(x)
    return f"Result for ur data -{result[0]}"

sample_input = {'Id': 1,
 'Store': 1,
 'DayOfWeek': 4,
 'Date': '2015-09-17 00:00:00',
 'Open': 1.0,
 'Promo': 1,
 'StateHoliday': '0',
 'SchoolHoliday': 0,
 'StoreType': 'c',
 'Assortment': 'a',
 'CompetitionDistance': 1270.0,
 'CompetitionOpenSinceMonth': 9.0,
 'CompetitionOpenSinceYear': 2008.0,
 'Promo2': 0,
 'Promo2SinceWeek': np.nan,
 'Promo2SinceYear': np.nan,
 'PromoInterval': np.nan}



print(user_prediction(sample_input))



# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
#
# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [20, 30, 40, 50],
#     "min_samples_leaf": [1, 2, 3, 4]
# }
#
# print("Starting")
# grid_search = GridSearchCV(model_r, param_grid, cv=TimeSeriesSplit(n_splits=3),scoring="neg_root_mean_squared_error", n_jobs=-1)
# start =time.time()
# grid_search.fit(x_train, target_train)
# result_time = (time.time() - start) / 60
# print(f"Result time to train model - {result_time} minutes")
# print(grid_search.best_params_)
#
# best_model = grid_search.best_estimator_
#
# val_prediction_r = best_model.predict(x_val)
#
# mae_r = mean_absolute_error(val_target,val_prediction_r)
# rmse_r = np.sqrt(mean_squared_error(val_target, val_prediction_r))
# rmspe_r = np.sqrt(np.mean(((val_target[no_null] - val_prediction_r[no_null]) / val_target[no_null]) **2))
#
# result = pd.DataFrame({
#     "Model": ["LinearRegression", "RandomForest"],
#     "mae": [mae, mae_r],
#     "rmse": [rmse, rmse_r],
#     "rmspe": [rmspe, rmspe_r]
# })
#
# print(result)