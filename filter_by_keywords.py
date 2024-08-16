import pandas as pd

# Load excel file
df = pd.read_excel("C:\\Users\\Chiara\\Desktop\\Thesis\\danish_news_sentences\\dan_news_2020_1M-sentences.xlsx")

# keywords to filter by
keywords = ['Hjemmearbejde', 'hjemmearbejde', 'hjemmekontoret', 'hjemmekontor', 'hjemmekontorene', 'hjemmekontorer', 'hjemmearbejdspladser', 'hjemmearbejdsdag']

# keywords specific for additional condition
specific_keywords = ['derhjemme', 'hjemmefra', 'Hjemmefra', 'hybrid', 'covid-19']

# Function to check if a sentence contains any of the specified keywords
def contains_keywords(sentence, specific_keywords):
   return any(keyword in sentence for keyword in specific_keywords)

#function to check if a sentence contains 'arbejde' or 'arbejder' or 'job'
def contains_arbejde(sentence):
   return "arbejde" in sentence or "arbejder" in sentence or "job" in sentence


# Boolean to filter rows
mask = df.apply(lambda column: any(keyword in column['Column2'] for keyword in keywords) or (contains_keywords(column['Column2'], specific_keywords) and contains_arbejde(column['Column2'])), axis=1)

# Filter df based on the mask
filtered_df = df[mask]

# Save to a new Excel file
filtered_df.to_excel('dan_news_2020_by_keywords.xlsx', index=False)
