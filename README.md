# IMMO ELIZA PROPERTY PRICE PREDICTION PROJECT

## Immo Eliza Data Analysis

### 1. **Description**

   This project is created as a consolidation project inside the group webscrape project 'challenge data analysis'.  
   It is a learning project for students at BeCode.org for the AI bootcamp.  
   The aim is to learn how to clean, analyze, and visually output the data that was retrieved via the web scrape described in the challenge data analysis repo.  
   Instead of working as a group, the analysis part and creation of the pipeline to predict property prices were made a solo challenge.  
   In this repo, you will find a Jupyter notebook containing all steps I made to visually present my conclusions on the dataset and how the creation of the pipeline to predict the prices was created.  

### 2. **Installation**

   Python version number: 3.11  

   Use a Jupyter notebook reader, I recommend Visual Studio Code, but any Jupyter notebook reader will do.  

   Setup a virtual environment and install the required libraries using this command in terminal, while you are in the main folder of this repo:  

   ```python
   pip install -r requirements.txt
   ```

### 3. **Workflow on this project**

#### 3.1 **Data cleaning**

   Normally, the first step would be to take a cleaned dataset and iron it to match our needs.  
   But in this case, I'm using the scraped data from the group project.  
   So I did end up spending some time to directly clean the data specifically for the prediction model.  
   This process can be found in the Immo-Eliza-Data-Analysis Jupyter notebook.  
   In the end, a heatmap showing the correlation between the most relevant features in the dataset was achieved.  

   ![Alt text](<data-exploration/Correlation Matrix.png>)

#### 3.2 **Prediction model**

   Using the cleaned data I started off to build a basic pipeline and training a LinearRegression model on the largest possible selection of features in the dataframe.  
   Next up, I learned it will be necessary to create a price prediction model based on only the relevant features.  
   The basic LinearRegression model didn't seem to achieve high scores and low MSE, so the journey into the non-linear regression models began.  
   After some trial and error, the XGBoost model came out on top.  

#### 3.3 **Deployment**

   Now that we have a working model, it is time to deploy:  
      -Pickle: Since we should not keep re-training the model every time it is used to predict a price, I will store the model in a .pkl file. For this I will use the pickle library.  
      -FastAPI: Now it is time to set up the app.py file to get an API (application programming interface) up and running locally with uvicorn.  
      -Docker: We have tested the get and post request using a local API, it is time to put the whole package together in a docker file. This way, the Immo-Elizza price predictor can be rendered anywhere.  
      -Render: As a final touch, the docker file is rendered on  [Render.com](https://render.com/)  

## 4. **How to use**

   The notebooks are created to be able to achieve a step by step insight on how the data analysis and the model(s) creation came to be.  

### 4.1 **For the model pipeline:**

   I have made a 2 optional arguments for the pipeline:  

   The syntax for the model pipeline is:  

   ```python
      model(df_input, model=1)
   ```

#### Explanation for the syntax

- `df_choice`: Optional integer input (1 = df_cleaned, 2 = df_house, 3 = df_apt) to select the dataframe to use. Standard option = 1  
- `model`: Optional input for model number (1 = XGBoost, 2 = LinearRegression) Standard option = 1  

Calling the script from main.py will ask for user input to set the choice of model and the dataframe to be used.  

Variables returned:  

- `score`: the score the model gets  
- `mse`: the mean squared error value  
- `cv_scores`: cross-validation R² score value  
- `mean_cv_score`: mean cross-validation R² score  
- `std_cv_score`: Standard deviation of cross-validation R2 scores  

Simply run the main.py file and the script for the pipeline will commence.  
The output will come as an informational printout, and a scatterplot will be generated to visualize the models results.  
Since we want to deploy an API, the model, scaler and encoder (OneHotEncoder) are saved in the /models folder as pickle/joblib files.  

![Alt text](<output/XGB model score.png>)

### 4.2 **For the API:**
  
   To run the FastAPI, navigate to this repo's root folder in your terminal and enter the following command:  

```bash
uvicorn app:app --reload
```  

   The code for the FastAPI can be found in app.py. (root folder)  
   The file calls the functions for the preprocessing and prediction in the src folder.  
  
   For the API the choice was made to run with XGBoost model and the df_clean DataFrame.  
   This combination got the best results in predicting the price.  
   The root folder (get '/') will return a string of requirements.  
   To receive a price prediction u can:  

- Go to the /predict/ folder and send a curl POST request with the json dictionary in the format explained in the root. (or use an app like Postman)  
- Navigate to the /docs page and click on the predict function and then try out. Fill in valid values for the json dictionary and click execute.  

## 4.3 **Dockerfile:**

The dockerfile to create the image is included in the root folder and is simply called: dockerfile  
To create an image using docker:  
    - Be sure to have docker installed on your system.
    - In your terminal, navigate to the root folder of this repo and enter the following command:

```bash
docker build . -t property_price_prediction
```  

You are free to change the name of the docker image to whatever suits you.
Wait for the image to be built and then create a container with the image.  
Use this command to run the container.

```bash
docker run property_price_prediction
```  

## 4.4 **Render:**

<https://immo-property-price-prediction.onrender.com/>  

The render is online on a free subscription so if link does not work, it is offline...

## 4.5 **Airflow Pipeline:**

The airflow pipeline is contained in the docker-airflow folder.
To run the docker containers and start the pipeline:

Download the folder to your device, navigate to the airflow-docker folder in your terminal and run this command:
Don't forget to run the docker software! (eg windows: Docker desktop)

```bash
docker compose up --build
```

once the image is up and running, open your browser and navigate to <http://localhost:8080>

```bash
login: BeCode
password: InCodeWeTrust
```

You should be able to find these credentials in the docker-compose.yml file and adapt it to your needs.

## 5.**Timeline**

### Part 1 of this project was started on 04/07/2023 09:00 and had a deadline set for 11/07/2023 12:30 (4.5 days)

- Be able to use `pandas`.
- Be able to use Data visualization libraries.(`matplotlib` or `seaborn`).
- Be able to clean a dataset for analysis.
- Be able to use colors in visualizations correctly.
- Be able to establish conclusions about a dataset.
- Be able to find and answer creative questions about data.
- Be able to think outside the box.

### Part 2 of this project was started on 17/07/2023 09:00 and had a deadline set for 20/07/2023 16:30 (4 days)

## Learning objectives

- Be able to apply a regression in a real context.
- Be able to preprocess data for machine learning.
- Be able to analyze the results of a machine learning model.
- You have to handle NaNs.
- You have to handle categorical data.
- You have to select features and preprocess them as needed.
- You have to remove features that have too strong correlation.
- You have to evaluate your models performance with an appropriate metric

### Part 3 of this project was started on 26/07/2023 09:00 and had a deadline set for 28/07/2023 16:00 (3 days)

## Mission objectives

- Be able to deploy a machine learning model.
- Be able to create an API that can handle a machine learning model.
- Deploy an API to Render with Docker.

### Part 4 of this project was started on 11/09/2023 09:00 and had a deadline set for 22/09/2023 16:00 (14 days)

## Mission objectives

- Scrape every night all the apartments on sale.
- Scrape every night all the houses on sale.
- Make an interactive dashboard to analyze the market.
- Train a regression model and evaluate its performance.

And the requirements:

- Your datasets need to be versioned.
- Your models need to be versioned.
- You will need to apply a different pre-processing for analyzing data and for training models.

## 6.**Personal notes**

   This project was a very big step into the world of data analysis for me.
   The data analysis for this project was overwhelming at first, but I got better at it, once I started using the data as features for my model.
   I managed to create a model using simple features from the data.
   In the near future, I will try to add some more features from using sklearn's OneHotEncoder, pandas get_dummies(), or even external sources.
   For now, I'm happy with the achievements I made this far.

   **edit after final week:**  

   When finishing this project, I did manage to use the OneHotEncoder on the categorical data used for the model training.  
   Also, a MinMaxScaler is applied to improve the model's accuracy.

## 7. **Thank you note**

   I would like to thank all fellow students and the coaches Vanessa and Sam at BeCode.org for any and all helpful information they gave me on this journey.

   A special thanks goes out to Nikolaas Willaert who is a co-student @BeCode.org:

   For his time and efforts to sit and debug with me. I learned a lot of tips and tricks from him. (and have some simmilar code 😉😉 )  

   <a href="https://github.com/nikolaaswillaert/" target="_blank">Github link for Nikolaas</a>.
