import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats

# load dataset into dataframe
data = pd.read_csv('tiktok_dataset.csv')

#display first 10 rows
print(data.head(10))

# generate table of descriptive statistics about the data
print(data.describe())

# check for missing values
print('missing values: ', data.isna())
print('sum of missing values: ', data.isna().sum())

# drop rows with missing values
data=data.dropna()

print('sum of missing values: ', data.isna().sum())

# Compute the mean video view count for each group in verified status
print('Mean view count: ', data.groupby('verified_status')['video_view_count'].mean())

# Hypothesis testing
#𝐻0: There is no difference in number of views between TikTok videos posted by verified accounts and TikTok videos posted by unverified accounts (any observed difference in the sample data is due to chance or sampling variability).
#𝐻𝐴: There is a difference in number of views between TikTok videos posted by verified accounts and TikTok videos posted by unverified accounts (any observed difference in the sample data is due to an actual difference in the corresponding population means).

# Conduct a two sample t-test to compare means

not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

print(stats.ttest_ind(a=not_verified, b=verified, equal_var=False))

# Since the p-value is extremely small, we reject the null hypothesis.
# In other words, there is a statistically significant difference in the mean video view count betweeen verified and unverified accounts on tiktok.





