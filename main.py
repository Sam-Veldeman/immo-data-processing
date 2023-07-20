from src.clean_data import run_cleanup
from src.linear_model import model
# Get user input for model_number
while True:
    model_input = input("Choose the model (1 for Linear Regression, 2 for XGBoost): ")
    if model_input.isdigit():
        model_number = int(model_input)
        if model_number in [1, 2]:
            break
    print("Invalid input. Please enter 1 or 2.")

# Get user input for scaled
while True:
    scale_input = input("Do you want to scale the data? (y/n): ")
    if scale_input.lower() in ['y', 'n']:
        scaled = True if scale_input.lower() == 'y' else False
        break
    print("Invalid input. Please enter 'y' or 'n'.")

# Get user input for DataFrame choice
while True:
    df_input = input("Choose the DataFrame (1 for entire DataFrame, 2 for houses, 3 for apartments): ")
    if df_input.isdigit():
        df_choice = int(df_input)
        if df_choice in [1, 2, 3]:
            break
    print("Invalid input. Please enter 1, 2, or 3.")


regressor, score, mse, cv_scores, mean_cv_score, std_cv_score, fig = model(df_input,model=model_number, scaled=scaled )
print(f'The score for the model is: {score}\nThe MSE for this model is {mse}')
print("Cross-validation R2 scores:", cv_scores)
print("Mean cross-validation R2 score:", mean_cv_score)
print("Standard deviation of cross-validation R2 scores:", std_cv_score)
fig.show()