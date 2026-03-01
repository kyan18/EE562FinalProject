import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

reviews = pd.read_json("yelp_training_set/yelp_training_set_review.json", lines=True)
users = pd.read_json("yelp_training_set/yelp_training_set_user.json", lines=True)
business = pd.read_json("yelp_training_set/yelp_training_set_business.json", lines=True)
check_in = pd.read_json("yelp_training_set/yelp_training_set_checkin.json", lines=True)

reviews['total_votes'] = reviews['votes'].apply(lambda x: x['funny'] + x['useful'] + x['cool'])
reviews['review_length'] = reviews['text'].apply(len)
reviews['date'] = pd.to_datetime(reviews['date'])
current_date = reviews['date'].max()
reviews['review_age_days'] = (current_date - reviews['date']).dt.days

merged = reviews.merge(
    users[['user_id', 'review_count', 'average_stars']],
    on='user_id',
    how='left'
)

merged = merged.rename(columns={
    'review_count': 'user_review_count',
    'average_stars': 'user_avg_stars'
})

merged = merged.merge(
    business[['business_id', 'review_count', 'stars']],
    on='business_id',
    how='left',
    suffixes=('', '_business')
)

merged = merged.rename(columns={
    'review_count': 'business_review_count',
    'stars': 'business_avg_stars'
})

# Distributions
def plot_auto_bar_distribution(series, title, xlabel, percentile=99):
    # Limit x-axis to percentile to focus on majority
    max_val = int(np.percentile(series, percentile))
    series_limited = series[series <= max_val]
    
    counts = series_limited.value_counts().sort_index()
    df = pd.DataFrame({'value': counts.index, 'count': counts.values})
    
    plt.figure(figsize=(15,5))
    sns.barplot(x='value', y='count', data=df, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel("Number of Entries")
    plt.title(title)
    
    step = max(1, len(df)//50)
    plt.xticks(np.arange(df['value'].min(), df['value'].max()+1, step), rotation=90)
    
    plt.ylim(0, df['count'].max()*1.05)
    
    plt.show()

# Average stars given by user
plot_auto_bar_distribution(users['average_stars'], "Distribution of Average Stars Given by User", "Average Stars Given by User")
# Total Votes
plot_auto_bar_distribution(reviews['total_votes'], "Distribution of Total Votes (Funny + Useful + Cool)", "Total Votes")
# Review Length
plot_auto_bar_distribution(reviews['review_length'], "Distribution of Review Length", "Review Length (chars)")
# User Review Counts
plot_auto_bar_distribution(users['review_count'], "Distribution of User Review Counts", "User Review Count")
# Business Review Counts
plot_auto_bar_distribution(business['review_count'], "Distribution of Business Review Counts", "Business Review Count")

def plot_category_distribution(series, title):
    counts = series.value_counts().head(20)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x=counts.values, y=counts.index)
    plt.xlabel("Number of Businesses")
    plt.ylabel("Category")
    plt.title(title)
    plt.show()

# Business type
business_exploded = business.explode('categories')

plot_category_distribution(
    business_exploded['categories'], 
    "Top 20 Business Categories"
)

# Correlation analysis
features = merged[
    [
        'user_review_count',
        'user_avg_stars',
        'business_avg_stars',
        'business_review_count',
        'review_length',
        'review_age_days',
        'total_votes'
    ]
]

corr_matrix = features.corr()

print("Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(6, 6))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm",
    annot_kws={"size":12},
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# Scatter plot
def plot_scatter(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Review length versus total votes
plot_scatter(merged['review_length'], merged['total_votes'], 'Review Length', 'Total Votes', 'Review Length vs Total Votes')
# User count versus total votes
plot_scatter(merged['user_review_count'], merged['total_votes'], 'User Review Count', 'Total Votes', 'User Review Count vs Total Votes')
# Business review count versus total votes
plot_scatter(merged['business_review_count'], merged['total_votes'], 'Business Review Count', 'Total Votes', 'Business Review Count vs Total Votes')

# Total check-ins 
rows = []
for _, row in check_in.iterrows():
    for key, val in row['checkin_info'].items():
        hour, day = map(int, key.split('-'))
        rows.append({'business_id': row['business_id'], 'hour': hour, 'day': day, 'checkins': val})

df_checkin = pd.DataFrame(rows)

weekday_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df_checkin['weekday'] = df_checkin['day'].map(weekday_map)

df_heatmap = df_checkin.groupby(['hour','weekday'])['checkins'].sum().reset_index()
df_heatmap = df_heatmap.pivot(index='hour', columns='weekday', values='checkins')
df_heatmap = df_heatmap[['Mon','Tue','Wed','Thu','Fri','Sat','Sun']]
df_heatmap = df_heatmap.sort_index()

plt.figure(figsize=(12,6))
sns.heatmap(df_heatmap, cmap='Oranges', annot=True, fmt=".0f", linewidths=0.5)
plt.title("Total Check-ins by Hour and Weekday")
plt.xlabel("Weekday")
plt.ylabel("Hour")
plt.tight_layout()
plt.show()
