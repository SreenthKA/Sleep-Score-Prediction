#!/usr/bin/env python
# coding: utf-8

# # DATA PREPROCESSING

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


sleep_stats_data = pd.read_csv(r"D:\SNU Chennai\projects\sleep score\sleep_stats.csv")
sleep_score_data = pd.read_csv(r"D:\SNU Chennai\projects\sleep score\sleep_score.csv").iloc[:,:2]


# i) preprocessing the sleep stats dataframe

# In[3]:


sleep_stats_data.head()


# In[4]:


sleep_stats_data.columns = sleep_stats_data.iloc[0]
sleep_stats_data.drop(sleep_stats_data.index[0], inplace=True)
sleep_stats_data


# In[5]:


sleep_stats_data.info()


# In[6]:


sleep_stats_data[sleep_stats_data['Minutes REM Sleep'].isna()]


# In[7]:


sleep_stats_data.dropna(axis=0, inplace=True)


# In[8]:


cols_to_convert = sleep_stats_data.columns[2:]
sleep_stats_data[cols_to_convert] = sleep_stats_data[cols_to_convert].astype(float)


# In[9]:


sleep_stats_data.info()


# ii) preprocessing the sleep score dataframe

# In[10]:


sleep_score_data.head()


# In[11]:


sleep_score_data.info()


# iii) joining both the dataframes

# In[12]:


sleep_stats_data.drop(columns='Start Time', inplace=True)
sleep_stats_data['Date'] = sleep_stats_data['End Time'].apply(lambda x: x[:10])
sleep_score_data['Date'] = sleep_score_data['timestamp'].apply(lambda x: x[:10])


# In[13]:


joined_data = sleep_stats_data.merge(sleep_score_data, on='Date', how='left')
joined_data.head()


# In[14]:


sleep_data = joined_data.drop(columns=['End Time', 'timestamp', 'Date', 'Number of Awakenings'])
sleep_data.dropna(axis=0,inplace=True)
sleep_data.head()


# In[15]:


len(sleep_data)


# # VISUALIZING THE RELATIONSHIPS BETWEEN FEATURES AND SLEEP SCORE

# i) Inspecting the relationship between all the independent variables and the dependent variables 

# In[16]:


def plot_relationships(df, num_cols):
    variables = df.columns
   
    dep_var = variables[-1]
    ind_var = variables[:-1]
    figs = len(dep_var)
    num_cols = num_cols
    num_rows = round(figs / num_cols) + 1
    fig = 1
    plt.figure(figsize=(20,30))
   
    for i in ind_var:
        pltfignums = [str(num_rows), str(num_cols), str(fig)]
        pltfig = int(''.join(pltfignums))
        plt.subplot(pltfig)
        plt.scatter(df[i], df[dep_var])
        plt.xlabel(str(i))
        plt.ylabel(str(dep_var))
        fig +=1


# In[17]:


plot_relationships(sleep_data, 3)


# ii) Inspecting the correlations

# In[18]:


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(sleep_data.corr(), annot=True, cmap='coolwarm', fmt='.2g')


# iii) Inspecting the sleep score distribution

# In[19]:


spread = int(max(sleep_data.overall_score) - min(sleep_data.overall_score))
spread


# In[20]:


plt.figure(figsize=(15,8))
plt.hist(sleep_data.overall_score, bins=spread)
plt.axvline(sleep_data.overall_score.mean(), color='r', label='Average Sleep Score')
plt.xlabel('Sleep Score')
plt.ylabel('Frequency')
plt.title('Sleep Score Distribution')
plt.legend()


# # Splitting the data into training, testing and validation set

# In[21]:


from sklearn.model_selection import train_test_split
X_train_temp, X_test, y_train_temp, y_test = train_test_split(sleep_data.iloc[:,:-1], sleep_data['overall_score'], test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=42)


# In[22]:


len(X_train), len(X_valid), len(X_test)


# # scaling the features

# In[23]:


plt.figure(figsize=(15,8))
sns.boxplot(data=X_train.iloc[:,1:], orient='h', palette='Set2')


# In[24]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = MinMaxScaler()
scaler.fit_transform(X_train)
scaler.transform(X_valid)
scaler.transform(X_test)


# # Feature selection using Lasso Regression

# In[25]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)


# In[26]:


lasso_coef = lasso.coef_

plt.figure(figsize=(10, 6))
plt.plot(range(len(X_train.columns)), lasso_coef)
plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=60)
plt.axhline(0.0, linestyle='--', color='r')
plt.ylabel('Coefficients')
plt.title('Lasso coefficients for sleep data features')


# In[27]:


cols_to_drop = ['Time in Bed', 'Minutes Light Sleep']

X_train_temp.drop(columns=cols_to_drop, inplace=True)
X_train.drop(columns=cols_to_drop, inplace=True)
X_valid.drop(columns=cols_to_drop, inplace=True)
X_test.drop(columns=cols_to_drop, inplace=True)


# # Defining the performance measure

# In[28]:


def scoring(model, test_features, test_labels):
    predictions = model.predict(test_features)
    mae = mean_absolute_error(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Mean Absolute Error: {:0.4f}.'.format(mae))
    print('Mean Squared Error: {:0.4f}.'.format(mse))
    print('R^2 Score = {:0.4f}.'.format(r2))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


# # Establishing a baseline

# In[29]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

baseline_y = [y_train.median()] * len(y_valid)

base_predictions = baseline_y
base_mae = mean_absolute_error(y_valid, base_predictions)
base_mse = mean_squared_error(y_valid, base_predictions)
base_r2 = r2_score(y_valid, base_predictions)
base_errors = abs(base_predictions - y_valid)
base_mape = 100 * np.mean(base_errors / y_valid)
base_accuracy = 100 - base_mape
print('Model Performance')
print('Mean Absolute Error: {:0.4f}.'.format(base_mae))
print('Mean Squared Error: {:0.4f}.'.format(base_mse))
print('R^2 Score = {:0.4f}.'.format(base_r2))
print('Accuracy = {:0.2f}%.'.format(base_accuracy))


# # 1) MLR - Multiple Linear Regression

# In[30]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
mlr = regressor.fit(X_train, y_train)


# In[31]:


scoring(mlr, X_valid, y_valid)


# In[32]:


y_pred = mlr.predict(X_valid)

x = np.linspace(65, 90, 25)
y = x

plt.scatter(y_pred, y_valid)
plt.plot(x, y)


# # 2) Random Forest Regression

# In[33]:


from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(random_state=42)
rf = rf_regressor.fit(X_train, y_train)


# In[34]:


scoring(rf, X_valid, y_valid)


# In[35]:


y_pred = rf.predict(X_valid)

x = np.linspace(65, 90, 25)
y = x

plt.scatter(y_pred, y_valid)
plt.plot(x, y)


# In[ ]:





# In[36]:


rf_feats = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['Feature Importance'])
rf_feats


# # 3) Extreme Gradiend Boosting

# In[37]:


from xgboost import XGBRegressor

xgb_regressor = XGBRegressor(random_state=42)
xgb = xgb_regressor.fit(X_train, y_train)


# In[38]:


scoring(xgb, X_valid, y_valid)


# In[39]:


y_pred = xgb.predict(X_valid)

x = np.linspace(65, 90, 25)
y = x

plt.scatter(y_pred, y_valid)
plt.plot(x, y)


# In[40]:


xgb_feats = pd.DataFrame(xgb.feature_importances_, index=X_train.columns, columns=['Feature Importance'])
xgb_feats


# # Cross-Validation

# In[ ]:


mlr_reg = LinearRegression()
rf_reg = RandomForestRegressor(random_state=42)
xgb_reg = xgb_regressor = XGBRegressor(random_state=42)


# In[ ]:


models = [mlr_reg, rf_reg, xgb_reg]


# In[ ]:


def cv_comparison(models, X, y, cv):
    
    cv_accuracies = pd.DataFrame()
    maes = []
    mses = []
    r2s = []
    accs = []
   
    for model in models:
        mae = -np.round(cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv), 4)
        maes.append(mae)
        mae_avg = round(mae.mean(), 4)
        mse = -np.round(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv), 4)
        mses.append(mse)
        mse_avg = round(mse.mean(), 4)
        r2 = np.round(cross_val_score(model, X, y, scoring='r2', cv=cv), 4)
        r2s.append(r2)
        r2_avg = round(r2.mean(), 4)
        acc = np.round((100 - (100 * (mae * len(X))) / sum(y)), 4)
        accs.append(acc)
        acc_avg = round(acc.mean(), 4)
        cv_accuracies[str(model)] = [mae_avg, mse_avg, r2_avg, acc_avg]
    cv_accuracies.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2', 'Accuracy']
    return cv_accuracies, maes, mses, r2s, accs


# In[ ]:


from sklearn.model_selection import cross_val_score
comp, maes, mses, r2s, accs = cv_comparison(models, X_train_temp, y_train_temp, 4)


# In[ ]:


comp.columns = ['Multiple Linear Regression', 'Random Forest', 'Extreme Gradient Boosting']
comp


# # Using the test data

# In[ ]:


mlr_final = LinearRegression()
rf_final = RandomForestRegressor(n_estimators = 200,min_samples_split = 6,min_impurity_decrease = 0.0,max_features = 'sqrt',max_depth = 25,criterion = 'absolute_error',bootstrap = True,random_state = 42)
xgb_final = XGBRegressor(tree_method = 'exact',objective = 'reg:squarederror',n_estimators = 1600,min_child_weight = 6,max_depth = 8,gamma = 0,eta = 0.1,random_state = 42)


# In[ ]:


mlr_final.fit(X_train_temp, y_train_temp)
rf_final.fit(X_train_temp, y_train_temp)
xgb_final.fit(X_train_temp, y_train_temp)


# In[ ]:


def final_comparison(models, test_features, test_labels):
    scores = pd.DataFrame()
    for model in models:
        predictions = model.predict(test_features)
        mae = round(mean_absolute_error(test_labels, predictions), 4)
        mse = round(mean_squared_error(test_labels, predictions), 4)
        r2 = round(r2_score(test_labels, predictions), 4)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = round(100 - mape, 4)
        scores[str(model)] = [mae, mse, r2, accuracy]
    scores.index = ['Mean Absolute Error', 'Mean Squared Error', 'R^2', 'Accuracy']
    return scores


# In[ ]:


final_scores = final_comparison([mlr_final, rf_final, xgb_final], X_test, y_test)
final_scores.columns  = ['Linear Regression', 'Random Forest', 'Extreme Gradient Boosting']
final_scores


# In[ ]:




