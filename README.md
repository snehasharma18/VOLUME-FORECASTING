# Volume-Forecasting
Volumes Forecasting

This Python script is a Streamlit web application for volume forecasting using the Prophet time series forecasting model. Here's a breakdown of what the code does:

Data Preparation:

It starts by reading data from an Excel file named 'Model Final Data.xlsx' and renames columns to 'ds' for the date and 'y' for the volume.
It also defines some variables, such as desired_rows, forecast_start_date, and forecast_end_date, to set up the forecasting horizon.
Creates a 'future' DataFrame with limited dates, regions, subprocesses, and functions for forecasting.
Data Cleaning:

Copies the original data into the 'df' DataFrame.
Converts the 'ds' column to a datetime format.
Checks for missing values in the DataFrame and prints a summary of the data.
Data Preprocessing:

Drops irrelevant columns ("Core", "Company", "Active Status") from the DataFrame.
Defines the training and test data based on the forecast start and end dates.
Modeling:

Initializes a Prophet model and fits it to the training data.
Generates forecasts using the model and stores the predictions in the 'forecast' DataFrame.
Computes mean absolute error (MAE) and root mean squared error (RMSE) as evaluation metrics for the forecast.
Streamlit App:

Sets up a Streamlit web application with a title and user input widgets for selecting a region, subprocess, function, year, and month.
When the "Predict" button is clicked, the script re-trains the Prophet model on the selected region, subprocess, and function.
Generates and displays forecasted values in a table and provides a downloadable CSV file.
Creates an interactive Plotly line chart showing the training and predicted data.
Overall, this code allows users to interactively select different forecasting scenarios and visualize the forecasted results, providing a user-friendly interface for volume forecasting based on historical data.
