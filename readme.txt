Final stage:
Tasks:
Billy has his data refined. It’s ready now to predict his marks based on his attributes. 
1.	Create a linear regression model by using sklearn.linear_model library. 
2.	Fit the model with the x_train and x_test values you had previously created.
3.	Calculate the accuracy by using the score attribute of the linear regression model on the x_test and y_test values.
4.	Using it’s predict attribute predict the values of all x_test, and then plot a scatter plot between the true and predicted values of x_test.
We don’t need all the features of the input, some features affect the accuracy a lot while some don’t affect it at all. Now we will be using backward elimination method to find the most important attributes which affect our prediction from that dataset.
5.	Using statsmodel.api’s OLS(Ordinary Least Squares) find how each feature affects the prediction and the model. Then based on that eliminate the features and keep on doing so till you achieve highest accuracy with suitable features are required.
