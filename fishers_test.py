import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, kendalltau, fisher_exact
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kendalltau

data = {'Label': ['positive', 'negative'],
        '2019': [6, 5],
        '2020': [48, 62],
        '2021': [24, 22],
        '2022': [18, 20]}

''' if contingency table also includes neutral label
data = {'Label': ['positive', 'negative', 'neutral'],
        '2019': [6, 5, 5],
        '2020': [48, 62, 176],
        '2021': [24, 22, 43],
        '2022': [18, 20, 36]}
'''

# pairs of years to compare
year_pairs = [('2019', '2020'), ('2019', '2021'), ('2019', '2022'), ('2020', '2021'), ('2020', '2022'), ('2021', '2022')]

# loop through each pair of years
for year_pair in year_pairs:
    year1, year2 = year_pair
    
    # create a contingency table for the two years
    contingency_table = [
        [data[year1][i], data[year2][i]] for i in range(len(data['Label']))
    ]
    
    # Run Fisher's Exact Test on each pair of sentiment comparisons within the years
    for i, sentiment1 in enumerate(data['Label']):
        for j, sentiment2 in enumerate(data['Label']):
            if i < j:  # no duplicate comparisons
                table = [
                    [contingency_table[i][0], contingency_table[j][0]],
                    [contingency_table[i][1], contingency_table[j][1]]
                ]
                odds_ratio, p_value = fisher_exact(table)
                print(f'Comparison between {year1} and {year2}: {sentiment1} vs {sentiment2}')
                print(f'Odds Ratio: {odds_ratio}')
                print(f'P-Value: {p_value}\n')
