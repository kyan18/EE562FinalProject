import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reviews = pd.read_json("yelp_training_set/yelp_training_set_review.json", lines=True)
users = pd.read_json("yelp_training_set/yelp_training_set_user.json", lines=True)
business = pd.read_json("yelp_training_set/yelp_training_set_business.json", lines=True)

reviews['useful'] = reviews['votes'].apply(lambda x: x['useful'])

# TODO: Change for new features
train_data = reviews.merge(
    business[['business_id', 'review_count', 'stars']], 
    on='business_id',
    how='left'
).dropna()
# train_data = reviews.copy()
# train_data['text_word_count'] = train_data['text'].apply(lambda x: len(x.split()))

# train_data['date'] = pd.to_datetime(train_data['date'])
# train_data['year'] = train_data['date'].dt.year
# train_data['month'] = train_data['date'].dt.month
# train_data['weekday'] = train_data['date'].dt.weekday

# train_data['days_since'] = (pd.Timestamp.now() - train_data['date']).dt.days

X_train = train_data[['review_count', 'stars_y']] # TODO: Change
y_train = train_data['useful']

test_reviews = pd.read_json("yelp_test_set/yelp_test_set_review.json", lines=True)
test_users = pd.read_json("yelp_test_set/yelp_test_set_user.json", lines=True)
test_business = pd.read_json("yelp_test_set/yelp_test_set_business.json", lines=True)

# TODO: Change
X_test = test_reviews.merge(
    test_business[['business_id', 'review_count', 'stars']],
    on='business_id',
    how='left'
).dropna()[['review_count', 'stars_y']]
# test_data = test_reviews.copy()
# test_data['text_word_count'] = test_data['text'].apply(lambda x: len(x.split()))

# test_data['date'] = pd.to_datetime(test_data['date'])
# test_data['year'] = test_data['date'].dt.year
# test_data['month'] = test_data['date'].dt.month
# test_data['weekday'] = test_data['date'].dt.weekday

# test_data['days_since'] = (pd.Timestamp.now() - test_data['date']).dt.days

# X_test = test_data[['text_word_count']]

regr = LinearRegression()
regr.fit(X_train, y_train)

train_preds = regr.predict(X_train)

print("MSE:", mean_squared_error(y_train, train_preds))
print("R²:", r2_score(y_train, train_preds))

# TODO: Change
print("review_count:", regr.coef_[0])
print("stars_y:", regr.coef_[1])

# print("year:", regr.coef_[1])
# print("month:", regr.coef_[2])
# print("weekday:", regr.coef_[3])

print("Intercept:", regr.intercept_)