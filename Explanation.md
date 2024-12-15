The goal of this study is to predict the number of strikeouts of a pitcher during a season using MLB data imported from www.baseballsavant.mlb.com, we're using data from 2024.
We'll use a Decision Tree Regressor as the model used to make the predictions. 

First, we load that data stored as a CSV file (stats.csv) using pandas to visualize the dataframe.
Then we drop the columns 'player_id' and 'year' since we won't take them into account for the regression model.
For our y value (target column), we'll use the 'strikeout' column.
For our X values we'll use the rest of the columns, except the 'strikeout' and 'last_name, first_name' columns. 
We use the method train_test_split from sklearn to split our data, our test_size will be 20% of the total data.
Then, we import the model from sklearn and fit our splitted data into it. 
We make the predictions on the test dataset.
We then plot the tree with a max_depth = 4, in order to better visualize the leaves. 
Perhaps this is the most interesting part of the study since we use the method feature_importance_ from the model in order to see which features have the most influence over the results.
whiff_percent and total strikes (p_total_strike) are the most important features, when it comes to predict the total number of strikeouts of a pitcher throughout a season.
