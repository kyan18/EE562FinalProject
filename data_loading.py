import pandas as pd

reviews = pd.read_json("C:\\Users\\kaito\\Downloads\\yelp-recruiting\\yelp_training_set\\yelp_training_set\\yelp_training_set_review.json", lines=True)
users = pd.read_json("C:\\Users\\kaito\\Downloads\\yelp-recruiting\\yelp_training_set\\yelp_training_set\\yelp_training_set_user.json", lines=True)
business = pd.read_json("C:\\Users\\kaito\\Downloads\\yelp-recruiting\\yelp_training_set\\yelp_training_set\\yelp_training_set_business.json", lines=True)
checkin = pd.read_json("C:\\Users\\kaito\\Downloads\\yelp-recruiting\\yelp_training_set\\yelp_training_set\\yelp_training_set_checkin.json", lines=True)

print("Loaded!")
print(reviews.shape)
# if all data is loaded, it should print a shape of (229907, 8)

reviews['useful_votes'] = reviews['votes'].apply(lambda x: x['useful'])

reviews = reviews.drop(columns=['votes'])

reviews['review_length'] = reviews['text'].apply(len)