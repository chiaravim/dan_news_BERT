import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

#database of classified sentences
df = pd.read_csv("C:\\Users\\Chiara\\Desktop\\Thesis\\Scripts\\hjemmearbejde_classification.csv")

#descriptive statistics

# convert 'Date'to datetime and strip whitespace
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d/%m/%Y')  # date format dd/mm/yyyy

# Descriptive statistics
descriptive_stats = df['Score'].describe()
print("Descriptive Statistics for Score column:")
print(descriptive_stats)

# extract the year 
df['Year'] = df['Date'].dt.year

# calculate descriptive statistics for each year
for year in range(2019, 2023):
    yearly_stats = df[df['Year'] == year]['Score'].describe()
    print(f"\nDescriptive Statistics for Score column in {year}:")
    print(yearly_stats)
# Convert dates to numerical
df['Date_numeric'] = (df['Date'] - df['Date'].min()).dt.days

#descriptive statistics for each label
descriptive_stats = df['Score'].describe()
print("Descriptive Statistics for the entire Score column:")
print(descriptive_stats)

# Calculate and print descriptive statistics for each label
labels = ['positive', 'negative', 'neutral']

for label in labels:
    label_stats = df[df['Label'] == label]['Score'].describe()
    print(f"\nDescriptive Statistics for Score column with label '{label}':")
    print(label_stats)

# Visualisations

# extracts year from the "Date" column
df['Year'] = pd.to_datetime(df['Date']).dt.year

label_mapping = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
df['Label'] = df['Label'].map(label_mapping)

# Group the data by the "Label" column and calculate the count of each label
label_counts = df['Label'].value_counts()

# label colours and markers
label_order = ['Negative', 'Neutral', 'Positive']
label_colors = {'Negative': 'tab:blue', 'Neutral': 'tab:orange', 'Positive': 'tab:green'}
#label_markers = {'Negative': '-', 'Neutral': 'o', 'Positive': '+'}

#scatterplot: date on x axis, confidence score on y axis and three labels shown in the graph as markers
plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x='Date', y='Score', hue='Label', style='Label', palette=label_colors, s=100)

plt.title('Confidence Score Over Time With Labels')
plt.xlabel('Date')
plt.ylabel('Confidence Score')
plt.legend(title='Sentiment Label')
plt.grid(True)
plt.show()

# pie chart plot
plt.figure(figsize=(8, 6))
plt.pie(label_counts[label_order], labels=label_counts[label_order].index, autopct='%1.1f%%', startangle=140, colors=[label_colors[label] for label in label_order])
plt.title('Overall distribution of Sentiment Labels')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Group the data by year and label, and calculate the count of each combination
label_counts_by_year = df.groupby(['Year', 'Label']).size().unstack(fill_value=0)
print(label_counts_by_year)
# Plot a grouped bar chart
label_counts_by_year.plot(kind='bar', figsize=(10, 6))
plt.title('Sentiment Labels by Year')
plt.xlabel('Year')
plt.ylabel('Sentence Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment Label')
plt.show()
