Attempting to improve on existing reaction yield prediction models by modifying the scope of the model prediction and the structural representation of chemical reactions.
-

Training data is taken from the USPTO patent dataset, as per the state of the art literature example: 	Digital Discovery, 2022,1, 91-97

This model uses and XGBRegressor algorithm with early stopping, trained on 6039 amide-bond forming reations between carboxylic acids and primary amines with 10 fold cross validation. The model uses both ECFP (3) fingerprints and DRFP fingerprints (from the cited article above) to describe the structural features of the reaction.

The model performance is scored by 4 metrics; the r-squared value to show the correlation between experimental and predicted yield, the root-mean-square-error which is the average error between the experimental and predicted yields and the % of results within 10 and 5% of the experimental yield.

Plots for each CV output can be found in the respective results folders.

This code requires the following python packages: drfp, scikit-learn, pandas, xgboost and numpy.
  
