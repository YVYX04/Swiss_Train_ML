# Forecasting Delays @ SBB
## Problem Statement 

© 2025, 2026 Yvan Richard.  
All rights reserved.

### Framing the Machine Learning Task

Now that I have a ready to use data set, I must settle on which specific task I will tackle and within which framework I will accomplish it. Here are the principle I will follow during this short project:

+ **Classification**. I will perform a classification task instead of a regression task. My goal is to predict the variable $y := $ `is_delayed`. This means that my ML project tackles a *binary classification task*. SBB considers that a train is delayed if it has a $3$ minutes delay compared to scheduled arrival. I encoded this accordingly in my data.

+ **Time-Dependency**. Predictions are made using only information that is realistically available before the scheduled arrival, including temporal context (time of day, day of week), network structure (station, line, service type), and lagged congestion indicators derived from prior train movements, thereby avoiding any form of data leakage.

+ **Train-validation-test Split**.
To ensure a leakage-free and realistic evaluation, the dataset is split strictly along the time dimension. All observations from January 1 to January 24 are used for model training, while the final week of January (January 25–31) is reserved for validation and hyperparameter tuning. The model is then evaluated on a completely held-out test set consisting of all observations from September 2025. This split prevents any information from the test period from influencing model selection, enforces a realistic forecasting setting, and allows assessment of the model’s ability to generalise across both time and seasonal regimes.