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
def plot_auto_bar_distribution(series, title, xlabel, percentile=99, bin_width=1):
    
    max_val = int(np.percentile(series, percentile))
    series_limited = series[series <= max_val]

    # Create bins
    bins = np.arange(0, max_val + bin_width + 1, bin_width)
    counts, edges = np.histogram(series_limited, bins=bins)

    # Compute bar centers
    centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(15,6))
    bars = plt.bar(centers, counts, width=bin_width*0.9, color='skyblue')

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)

    # Set ticks at integer boundaries
    step = max(1, len(edges)//50)
    plt.xticks(edges[::step], rotation=45)

    plt.ylim(0, counts.max()*1.1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 height,
                 f'{int(height)}',
                 ha='center',
                 va='bottom',
                 fontsize=8)

    plt.show()

# Average stars given by user
plot_auto_bar_distribution(users['average_stars'], "Distribution of Average Stars Given by User", "Average Stars Given by User", bin_width=1)
# Total Votes
plot_auto_bar_distribution(reviews['total_votes'], "Distribution of Total Votes (Funny + Useful + Cool)", "Total Votes", bin_width=1)
# Review Length
plot_auto_bar_distribution(reviews['review_length'], "Distribution of Review Length", "Review Length (chars)", bin_width=90)
# User Review Counts
plot_auto_bar_distribution(users['review_count'], "Distribution of User Review Counts", "User Review Count", bin_width=15)
# Business Review Counts
plot_auto_bar_distribution(business['review_count'], "Distribution of Business Review Counts", "Business Review Count", bin_width=5)

# Total check-ins 
rows = []
for _, row in check_in.iterrows():
    for key, val in row['checkin_info'].items():
        hour, day = map(int, key.split('-'))
        rows.append({'business_id': row['business_id'], 'hour': hour, 'day': day, 'checkins': val})

df_checkin = pd.DataFrame(rows)

# Heatmap of check-ins by hour and weekday
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

# Distribution of business categories
plot_category_distribution(
    business_exploded['categories'], 
    "Top 20 Business Categories"
)

# Correlation analysis between features
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
    cmap="BuPu",
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
    plt.scatter(
        x, y,
        s=5,
        alpha=0.2,
        edgecolors='none'
    )

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)

    plt.plot(x_clean, p(x_clean), linewidth=2, color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# User count versus total votes
plot_scatter(merged['user_review_count'], merged['total_votes'], 'User Review Count', 'Total Votes', 'User Review Count vs Total Votes')
# Review length versus total votes
plot_scatter(merged['review_length'], merged['total_votes'], 'Review Length', 'Total Votes', 'Review Length vs Total Votes')
# Review age versus total votes
plot_scatter(merged['review_age_days'], merged['total_votes'], 'Review Age (days)', 'Total Votes', 'Review Age vs Total Votes')
# Business review count versus total votes
plot_scatter(merged['business_review_count'], merged['total_votes'], 'Business Review Count', 'Total Votes', 'Business Review Count vs Total Votes')
