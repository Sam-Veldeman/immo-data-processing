# IMMO ELIZA PROJECT
This repo contains a jupyter notebook, use a viewer like visua

Immo Eliza Data Analysis
# 1. Description

This project is created as a consolidation project inside the group webscrape project 'challenge data analysis'. It is a learning project for students at BeCode.org for the AI bootcamp.
The aim is to learn how to clean, analyse and visualy output the data that was retrieved via the webscraper described in the challenge data analysis repo.
In stead of working as a group, the analysis part and creation of the pipeline to predict property prices was made a solo challenge.
In this repo, you will find a jupyter notebooks containing all steps I made to visually present my conclusions on the dataset and how the creation of the pipeline to prdedict the prices came to be.

# 2. Installation

Use a jupyter notebook reader, I recommend Visual Studio Code, but any jupyter notebook reader will do.
Setup a virtual environment and install the required libraries by using this command:

pip install -r requirements.txt

steps to realise the pipeline:

# 3. Workflow on this project

## 3.1 Data cleaning

Normally, the first step would be to take a cleaned dataset and iron it to match our needs.
But in this case, I'm using the scraped data from the group project.
So I did end up spending some time to directly clean the data specificaly for the prediction model.
This process can be found in the Immo-Eliza-Data-Analysis jupyter notebook.
In the end, a heatmap showing the correlation between the most relevant features in the dataset was achieved:

![Alt text](<data-exploration/Correlation Matrix.png>)


## 3.2 Prediction model

Using the cleaned data I started off to build a basice pipeline and training a LinearRegression model on the largest possibele selection of features in the dataframe.
Next up, I learned it will be necessary to create a price prediction model based on only the relevant features.
The basic LinearRegression model didn't seem to achieve high scores and low MSE, so the journey into the non-linear regression models began.
After some trial and error, the XGBoost model came out on top.

# 4. How to use

So the notebooks are created to be able to get a step by step insight on how the data analysis and the model(s) creation came to be.

For the model pipeline itself, I have made a few optional arguments for the pipeline:

The syntax for the model pipeline is 

```
model(df, columns, model=1, scaled=True)
```

Explantion for the syntax:

model(dataframe to use, list of columns to use as features, model number, scale X_train and X_test)

The last 2 arguments are optional. When not used, the model number will be set to 1, and scaling is set to True.

Available models:

1 = XGBoost
2 = LinearRegression

Calling the function will ask for user input to set choice of model and scaling the X_train, X_test.
Variables returned will be:

regressor: the model itself for future use.
score: the score the model gets
mse: the mean squared error value
cv_scores: cross validation R² score value
mean_cv_score: mean cross validation R² score
std_cv_score: Standard deviation of cross-validation R2 scores
fig: a usable variable 

The output will come as a informational printout and a scatterplot will be shown.

# 5. Timeline

Part 1 of this project was started on 04/07/2023 09u00 and had a deadline set for 11/07/2023 12:30 (4.5 days)
Part 2 of this project was started on 17/07/2023 09u00 and had a deadline set for 20/07/2023 16:30 (4 days)

# 6. Personal notes

This project was a verry big step into the world of data analyses for me.
The data analysis for this project was overwhelming at first, but I got better once I started using it as features.
I managed to create a model using simple featurs from the data.
In the near future I will try to add some more features from using sklearns OneHotEncoder, pandas get_dummies() or even external sources.
For now I'm happy with the achievements I made this far.

# 7. Thank you note

I would like to thank all fellow students and the coaches Vanessa and Sam at BeCode.org for any and all helpfull information they gave me in this journey.