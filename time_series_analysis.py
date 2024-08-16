import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, kendalltau
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kendalltau

#database of classified sentences
df = pd.read_csv("C:\\Users\\Chiara\\Desktop\\Thesis\\Scripts\\hjemmearbejde_classification.csv")

# convert 'Date' column to datetime and strip whitespace
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d/%m/%Y')  # date format dd/mm/yyyy

# group by day and sentiment label, then apply rolling average
daily_counts = df.groupby(['Date', 'Label']).size().unstack(fill_value=0)
#30day rolling count
rolling_counts = daily_counts.rolling(window=30, min_periods=1).mean()
#7day rolling count
#rolling_counts = daily_counts.rolling(window=7, min_periods=1).mean()

# Plot the rolling averages
rolling_counts.plot(kind='line', marker='.', markersize=3, linewidth=1)
plt.title('Rolling Averages (30-day window)')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# dictionary to store Tau and p-values
results = {}

# Mann-Kendall test for each sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    if sentiment in rolling_counts.columns:
        # time series data for sentiment
        sentiment_data = rolling_counts[sentiment]
        
        # use dates as the time component for the test
        time_index = pd.to_numeric(sentiment_data.index)
        
        # Mann-Kendall test
        tau, p_value = kendalltau(time_index, sentiment_data)
        results[sentiment] = {'tau': tau, 'p_value': p_value}

        # print Tau and p-value
        print(f'{sentiment.capitalize()} Sentiment:')
        print(f'  Tau: {tau:.3f}')
        print(f'  p-value: {p_value:.3f}')

        # Access p_value from results dictionary
        alpha = 0.05  # significance level
        if results[sentiment]['p_value'] < alpha:
            trend = 'increasing' if results[sentiment]['tau'] > 0 else 'decreasing'
            print(f'  There is a significant {trend} trend in {sentiment} sentiment.')
        else:
            print(f'  There is no significant trend in {sentiment} sentiment.')

    print()  # New line for better readability




















'''
# Group by time (daily)
df.set_index('Date', inplace=True)
daily_counts = df.groupby(['Date', 'Label']).size().unstack(fill_value=0)

# Apply a 30-day rolling average
rolling_counts = daily_counts.rolling(window=30, min_periods=1).mean()

# Plot the rolling averages
rolling_counts.plot(figsize=(12, 6))
plt.title('30-Day Rolling Average Sentiment Counts')
plt.show()

# Decompose the time series for each sentiment class
for sentiment in ['positive', 'negative', 'neutral']:
    result = seasonal_decompose(rolling_counts[sentiment].dropna(), model='additive', period=30)
    result.plot()
    plt.title(f'Time Series Decomposition - {sentiment.capitalize()} Sentiment')
    plt.show()

# Fit an ARIMA model (example for positive sentiment)
model = ARIMA(rolling_counts['positive'].dropna(), order=(1, 1, 1))
fit_model = model.fit()
print(fit_model.summary())

# Fit an ARIMA model (example for neutral sentiment)
model = ARIMA(rolling_counts['neutral'].dropna(), order=(1, 1, 1))
fit_model = model.fit()
print(fit_model.summary())


# Fit an ARIMA model (example for negative sentiment)
model = ARIMA(rolling_counts['negative'].dropna(), order=(1, 1, 1))
fit_model = model.fit()
print(fit_model.summary())

# Forecasting positive
forecast = fit_model.forecast(steps=12)  # Forecasting for the next 12 periods

# Plotting
plt.figure(figsize=(12, 6))

# Plot observed data, starting from 2018-01-01
observed = rolling_counts['positive']
plt.plot(observed, label='Observed')

# Handle the forecast period, extending the x-axis to cover 2018 onwards
# Creating a dummy period for the x-axis to start in 2018
full_index = pd.date_range(start='2018-01-01', end=observed.index[-1], freq='D')
observed_reindexed = observed.reindex(full_index).fillna(0)  # Fill with 0s or use NaN if preferred

# Now plot the forecast starting from the end of the observed data
forecast_index = pd.date_range(start=observed.index[-1], periods=12+1, freq='M')[1:]  # Adjust the forecast index
plt.plot(forecast_index, forecast, label='Forecast', linestyle='--')

plt.title('Positive Sentiment Forecast')
plt.legend()
plt.xlim(pd.Timestamp('2018-01-01'), forecast_index[-1])  # Set x-axis limits
plt.show()

# Forecasting neutral
forecast = fit_model.forecast(steps=12)  # Forecasting for the next 12 periods
plt.plot(rolling_counts['neutral'], label='Observed')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title('neutral Sentiment Forecast')
plt.legend()
plt.show()

# Forecasting negative
forecast = fit_model.forecast(steps=12)  # Forecasting for the next 12 periods
plt.plot(rolling_counts['negative'], label='Observed')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title('negative Sentiment Forecast')
plt.legend()
plt.show()

# Mann-Kendall Trend Test for positive sentiment
tau, p_value = kendalltau(rolling_counts.index, rolling_counts['positive'])
print(f'Mann-Kendall test p-value for positive sentiment: {p_value}')


# Mann-Kendall Trend Test for neutral sentiment
tau, p_value = kendalltau(rolling_counts.index, rolling_counts['neutral'])
print(f'Mann-Kendall test p-value for neutral sentiment: {p_value}')


# Mann-Kendall Trend Test for negative sentiment
tau, p_value = kendalltau(rolling_counts.index, rolling_counts['negative'])
print(f'Mann-Kendall test p-value for negative sentiment: {p_value}')

'''
'''
#group by time (monthly)
df['month'] = df['Date'].dt.to_period('M')
monthly_counts = df.groupby(['month', 'Label']).size().unstack(fill_value=0)


# Convert PeriodIndex back to DatetimeIndex for certain operations
monthly_counts.index = monthly_counts.index.to_timestamp()

full_index = pd.date_range(start=monthly_counts.index.min(), end=monthly_counts.index.max(), freq='M')
monthly_counts = monthly_counts.reindex(full_index, fill_value=0)

monthly_counts.plot(figsize=(12, 6))
plt.title('Monthly Sentiment Counts')
plt.show()

for sentiment in ['positive', 'negative', 'neutral']:
    result = seasonal_decompose(monthly_counts[sentiment], model='additive')
    result.plot()
    plt.title(f'Time Series Decomposition - {sentiment.capitalize()} Sentiment')
    plt.show()


# Fit an ARIMA model (example for positive sentiment)
model = ARIMA(monthly_counts['positive'], order=(1,1,1))
fit_model = model.fit()
print(fit_model.summary())

# Forecasting
forecast = fit_model.forecast(steps=12)  # Forecasting for the next 12 periods
plt.plot(monthly_counts['positive'], label='Observed')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title('Positive Sentiment Forecast')
plt.legend()
plt.show()

# Mann-Kendall Trend Test for positive sentiment
tau, p_value = kendalltau(monthly_counts.index.to_timestamp(), monthly_counts['positive'])
print(f'Mann-Kendall test p-value for positive sentiment: {p_value}')
'''