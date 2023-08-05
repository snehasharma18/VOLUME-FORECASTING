import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.express as px
import plotly.graph_objects as go

data=pd.read_excel('Model Final Data.xlsx')

data.rename(
    columns={"Month-Year":"ds", "Volume":"y"}, 
    inplace=True
)

data['Region'].value_counts()

regions = ['AMER','APAC','EMEA']
subprocess = ['E2C','IP']
function = ['Accounts Payable']

desired_rows = 6

forecast_start_date = pd.to_datetime('2023-05-01')  # Replace with your desired start date
forecast_end_date = forecast_start_date + pd.DateOffset(months=6)

future_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='M')
limited_dates = future_dates[:desired_rows]
limited_regions = (regions * (desired_rows // len(regions))) + regions[:desired_rows % len(regions)]
limited_subprocess = (subprocess * (desired_rows // len(subprocess))) + subprocess[:desired_rows % len(subprocess)]
limited_functions = (function * (desired_rows // len(function))) + function[:desired_rows % len(function)]

len(future_dates)

future = pd.DataFrame({
    'ds': limited_dates,
    'function' : limited_functions,
    'region': limited_regions,
    'subprocess': limited_subprocess
})

df=data.copy()

df.info()

df["ds"] = pd.to_datetime(df["ds"])

df.isnull().sum()

df.describe()

col_to_drop = ["Core","Company","Active Status"]
df = df.drop(col_to_drop, axis=1)

forecast_start_date = pd.to_datetime('2023-05-01')

forecast_end_date = forecast_start_date + pd.DateOffset(months=6)

train_data = df[df['ds'] < forecast_start_date][['ds', 'Region', 'Sub Process','Function','y']]
test_data = df[(df['ds'] >= forecast_start_date) & (df['ds'] <= forecast_end_date)]

model = Prophet()
model.fit(train_data)

forecast = model.predict(future)

forecast_with_region_subprocess = pd.concat([future, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]], axis=1)

predicted_values = forecast['yhat'][-90:]

actual_values = df[(df['ds'] >= forecast_start_date) & (df['ds'] <= forecast_end_date)]['y']

mae = mean_absolute_error(actual_values, predicted_values)
print("Mean Absolute Error:", mae)

rmse = mean_squared_error(actual_values, predicted_values, squared=False)
print("Root Mean Squared Error:", rmse)

st.title('Volume Forecasting App')

region = st.selectbox('Select Region', df['Region'].unique())
subprocess = st.selectbox('Select Subprocess', df['Sub Process'].unique())
function = st.selectbox('Select Function',df['Function'].unique())

current_year = pd.to_datetime('today').year

# Date input widgets for selecting future months
selected_year = st.selectbox('Select Year', range(2023, 2033))
selected_month = st.selectbox('Select Month', range(1, 13))

# Date for forecasting based on user selection
forecast_start_date = pd.to_datetime(f'{selected_year}-{selected_month:02d}-01').date()

predict_button = st.button("Predict")

if predict_button:
    # Convert forecast start date to datetime64[ns]
    forecast_start_date = pd.to_datetime(forecast_start_date)

    # Filter data for the selected region and subprocess
    filtered_data = df[(df['Region'] == region) & (df['Sub Process'] == subprocess) & (df['Function'] == function)]

    # Train the Prophet model on the filtered data
    train_start_date = forecast_start_date - pd.DateOffset(months=12)
    train_data = filtered_data[filtered_data['ds'] < forecast_start_date][['ds', 'y']]
    model = Prophet()
    model.fit(train_data)

    # Generate future dates for forecasting
    forecast_end_date = forecast_start_date + pd.DateOffset(months=6)
    future_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='M')
    future = pd.DataFrame({'ds': future_dates})

    # Generate forecasts using the trained model
    forecast = model.predict(future)

    # Display the forecasted values with 'ds' as date only and 'yhat' as integers
    st.write(forecast[['ds', 'yhat']].apply(lambda x: x.dt.date if x.name == 'ds' else x.astype(int) if x.name == 'yhat' else x, axis=0))
    
    csv_filename = 'predicted_values.csv'
    st.download_button('Download Predicted Values', forecast[['ds', 'yhat']].to_csv(index=False), file_name=csv_filename, mime='text/csv')
    
    # Create the Plotly line chart
    fig = go.Figure()

    # Add training data line
    fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'], mode='lines+markers', name='Training', line=dict(color='blue')))

    # Add predicted data line
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Predicted', line=dict(color='red')))

    # Customize the layout
    fig.update_layout(
        title='Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x',
        showlegend=True
    )

    # Display the line chart using Streamlit
    with st.container():
        st.plotly_chart(fig)



