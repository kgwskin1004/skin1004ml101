
# SKIN1004 ML 101

This repository contains practice code for the SKIN1004 ML 101 seminar. The example code consists of two problems:

1. Predict Amazon page view count.
2. Predict the influencer efficiency score.

## Pre-requisites
1. Python 3.9 or higher.
2. Install the required packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Problem 1: Predict Amazon Page View Count

The dataset is extracted from `amazon_detail_page_sales_and_traffic_by_asin`, focusing on the `Page_Views_Total` column. The goal is to predict the page view count based on past view counts. We'll employ LSTM and Transformer models for this task, with these models taking past view counts as input and predicting future view counts.

## Problem 2: Predict the Influencer Efficiency Score

For this task, we'll use the following model:

$$
a\log_e\left(1+ \frac{\text{view}}{\text{cost}}\right) + b\log_e\left(1 + \frac{\text{saved}}{\text{cost}}\right) + c\log_e\left(1+\frac{\text{comment}}{\text{cost}}\right) + d\log_e\left(1+\frac{\text{like}}{\text{cost}}\right) + e\log_e\left(1+\frac{\text{share}}{\text{cost}}\right)
$$

Specifically, we aim to predict the values of \( a \), \( b \), \( c \), \( d \), and \( e \). To achieve this, we'll use gradient descent.


## Let's practice! 🚀
1. List hyperparameters for each model.
2. Implement Transformer model for Problem 1. LSTM is implemented for your reference.
3. Apply the model implemented in Problem 1 to non-paid influencer data.

## Now, it's time to apply this for SKIN1004! 🌟
1. Predict the next month revenue based on the past page view count.
2. Predict the influencer efficiency score for new influencers.