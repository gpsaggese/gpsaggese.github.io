# Key Insights Report

- Average age of users: 44.22
- Correlation between session duration and monthly spend: -0.01
- Churn rates segmented by age and gender:
     age  gender  churn_rate
0     18  female    0.000000
1     18    male    0.250000
2     19  female    0.142857
3     19    male    0.428571
4     20  female    0.166667
..   ...     ...         ...
111   68  female    0.000000
112   68    male    0.500000
113   68   other    0.000000
114   69  female    0.250000
115   69    male    0.166667

[116 rows x 3 columns]
- Engagement metrics of premium vs non-premium users:
has_premium
0    10.733514
1    10.735385
Name: sessions_last_30d, dtype: float64
- Trends in spending behavior over time:
                   signup_days_ago  monthly_spend_usd
signup_days_ago           1.000000          -0.062433
monthly_spend_usd        -0.062433           1.000000
