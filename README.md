# IMMO ELIZA PROJECT

## Immo Eliza Data Analysis
1. **Description**

   This project is created as a consolidation project inside the group webscrape project 'challenge data analysis'.     
   It is a learning project for students at BeCode.org for the AI bootcamp.     
   The aim is to learn how to clean, analyze, and visually output the data that was retrieved via the webscraper described in the challenge data analysis repo.     
   Instead of working as a group, the analysis part and creation of the pipeline to predict property prices were made a solo challenge.     
   In this repo, you will find a Jupyter notebook containing all steps I made to visually present my conclusions on the dataset and how the creation of the pipeline to predict the prices came to be.

2. **Installation**

   Python version number: 3.11
   
   Use a Jupyter notebook reader, I recommend Visual Studio Code, but any Jupyter notebook reader will do. 
   
   Setup a virtual environment and install the required libraries by using this command in terminal while you are in the main folder of this repo:    
   ```
   pip install -r requirements.txt
   ```
3. **Workflow on this project**
   3.1 **Data cleaning**

   Normally, the first step would be to take a cleaned dataset and iron it to match our needs.      
   But in this case, I'm using the scraped data from the group project.     
   So I did end up spending some time to directly clean the data specifically for the prediction model.     
   This process can be found in the Immo-Eliza-Data-Analysis Jupyter notebook.     
   In the end, a heatmap showing the correlation between the most relevant features in the dataset was achieved.

   ![Alt text](<data-exploration/Correlation Matrix.png>)

   3.2 **Prediction model**

   Using the cleaned data I started off to build a basic pipeline and training a LinearRegression model on the largest possible selection of features in the dataframe.     
   Next up, I learned it will be necessary to create a price prediction model based on only the relevant features.     
   The basic LinearRegression model didn't seem to achieve high scores and low MSE, so the journey into the non-linear regression models began.     
   After some trial and error, the XGBoost model came out on top.

4. **How to use**

   The notebooks are created to be able to achieve a step by step insight on how the data analysis and the model(s) creation came to be.     
   For the model pipeline itself, I have made a few optional arguments for the pipeline:

   The syntax for the model pipeline is:

   ```
      model(df, columns, model=1, scaled=True)
   ```
   Explanation for the syntax:

   - `model`: dataframe to use
   - `columns`: list of columns to use as features
   - `model number`: model number (1 = XGBoost, 2 = LinearRegression)
   - `scale`: True if you want to scale X_train and X_test (optional)
   <br/><br/>
   Calling the script from main.py will ask for user input to set the choice of model and scaling the X_train, X_test. Variables returned will be:

   - `regressor`: the model itself for future use.
   - `score`: the score the model gets
   - `mse`: the mean squared error value
   - `cv_scores`: cross-validation R² score value
   - `mean_cv_score`: mean cross-validation R² score
   - `std_cv_score`: Standard deviation of cross-validation R2 scores
   - `fig`: a usable variable
   <br/><br/>
   Simply run the main.py file and the script for the pipeline will commence.     
   The output will come as an informational printout, and a scatterplot will be generated to visualize the models results.

5. **Timeline**

   Part 1 of this project was started on 04/07/2023 09:00 and had a deadline set for 11/07/2023 12:30 (4.5 days)

   Part 2 of this project was started on 17/07/2023 09:00 and had a deadline set for 20/07/2023 16:30 (4 days)

6. **Personal notes**

   This project was a very big step into the world of data analyses for me.     
   The data analysis for this project was overwhelming at first, but I got better once I started using it as features.     
   I managed to create a model using simple features from the data.     
   In the near future, I will try to add some more features from using sklearn's OneHotEncoder, pandas get_dummies(), or even external sources.     
   For now, I'm happy with the achievements I made this far.

7. **Thank you note**

   I would like to thank all fellow students and the coaches Vanessa and Sam at BeCode.org for any and all helpful information they gave me on this journey.
