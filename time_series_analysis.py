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
