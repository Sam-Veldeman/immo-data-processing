from src.clean_data import run_cleanup
from src.linear_model import model
from src.linear_model import get_model_input
from src.linear_model import get_scaled_input
from src.linear_model import get_df_input

# Get user inputs for model, scaling, and DataFrame choice
model_number = get_model_input()
scaled = get_scaled_input()
df_choice = get_df_input()

regressor, score, mse, cv_scores, mean_cv_score, std_cv_score, fig = model(df_choice,model=model_number, scaled=scaled )
print(f'The score for the model is: {score}\nThe MSE for this model is {mse}')
print("Cross-validation R2 scores:", cv_scores)
print("Mean cross-validation R2 score:", mean_cv_score)
print("Standard deviation of cross-validation R2 scores:", std_cv_score)
fig.show()