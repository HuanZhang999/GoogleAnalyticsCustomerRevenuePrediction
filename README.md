# Google-Analytics-Customer-Revenue-Prediction
https://www.kaggle.com/c/ga-customer-revenue-prediction

Introduction

The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. As such, marketing teams are challenged to make appropriate investments in promotional strategies.

This competition, which was organised by RStudio, Google cloud and Kaggle, challenged the attendants to analyze a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict revenue per customer, with the hope that the outcome would be more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

Datasets 

The dataset comprises a training set, containing user transactions from August 1st 2016 to April 30th 2018 and a test set, containing user transactions from May 1st 2018 to October 15th 2018. The submission file should be forward-looking predictions of PredictedLogRevenue for each of these fullVisitorIds for the timeframe of December 1st 2018 to January 31st 2019.

The columns in train and test datasets:
fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
channelGrouping - The channel via which the user came to the Store.
date - The date on which the user visited the Store.
device - The specifications for the device used to access the Store.
geoNetwork - This section contains information about the geography of the user.
socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
totals - This section contains aggregate values across the session.
trafficSource - This section contains information about the Traffic Source from which the session originated.
visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
visitNumber - The session number for this user. If this is the first session, then this is set to 1.
visitStartTime - The timestamp (expressed as POSIX time).
hits - This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.
customDimensions - This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.
totals - This set of columns mostly includes high-level aggregate data.
    
Why did I do this:

I translated the winner's (from Konstantin Nikolaev) R code to Python code while I found his strategy simple and effective. 

- Train creation: he took 4 non-overlapping windows of 168 days, calculated features for users in each period and calculated target for each user on each corresponding 62-day window. Then those 4 dataframes were combined in one train set.

- Problem as “classification and regression”: he predicted the probability of returning of customer amd the amount of transactions for those customers who returned seperetly.

The original solution and his R codes can be found here:
https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614#latest-482575
