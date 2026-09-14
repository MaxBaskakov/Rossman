# Rossmann Store Sales Prediction

This is my second AI/ML project. The task was to build a model that predicts sales based on data from Rossmann stores.

This is a **supervised regression problem** because the model is trained on labeled historical data, with `Sales` as the target variable. The target is a continuous numerical value.

I split the data into training and validation sets based on the `Date` column:

* **Training data:** 2013–2014
* **Validation data:** 2015

I cleaned the data and prepared it for regression. This included handling missing values, encoding categorical features, and scaling numerical features.

### Models

First, I trained a **Linear Regression** model and evaluated its performance using MAE, RMSE, and RMSPE.

Then, I trained a **Random Forest Regressor** and evaluated it using the same metrics.
### Old results
| Model             |        MAE |        RMSE |      RMSPE |
| :---------------- | ---------: | ----------: | ---------: |
| Linear Regression |    1699.06 |     2457.49 |     46.43% |
| **Random Forest** | **650.59** | **1051.79** | **17.05%** |

Based on these results, **Random Forest performed significantly better** than Linear Regression for this task.

### Hyperparameter Tuning

I also experimented with **GridSearchCV** to find better Random Forest hyperparameters.

However, the tuned model did not improve the results on the validation set. This was a useful experiment because it showed me that hyperparameter tuning does not always lead to better performance on unseen data.

### Prediction

I also implemented a `user_prediction()` function that allows the model to make predictions on new store data.

### Feature engineering 

I experimented with data and added feature engineering. The new features improved MAE and RMSE. RMSPE remained almost the same 

### New results 
| Model                 | MAE | RMSE | RMSPE |
|:----------------------| :---: | :---: | :---: |
| **Linear Regression** | 1706.25 | 2452.85 | 47.03% |
| **Random Forest**     | **627.02** | **1017.65** | **17.43%** |