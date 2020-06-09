import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv("C:/Users/PRUSTY/Downloads/DS_ML_FinalTask/DS_ML_FinalTask/student-math.csv", sep=";")

#Encoding all nominal and binary values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['school']= le.fit_transform(df['school'])
df['Pstatus']= le.fit_transform(df['Pstatus'])
df['sex']= le.fit_transform(df['sex'])
df['address']= le.fit_transform(df['address'])
df['famsize']= le.fit_transform(df['famsize'])
df['Mjob']= le.fit_transform(df['Mjob'])
df['Fjob']= le.fit_transform(df['Fjob'])
df['reason']= le.fit_transform(df['reason'])
df['guardian']= le.fit_transform(df['guardian'])
df['schoolsup']= le.fit_transform(df['schoolsup'])
df['famsup']= le.fit_transform(df['famsup'])
df['paid']= le.fit_transform(df['paid'])
df['activities']= le.fit_transform(df['activities'])
df['nursery']= le.fit_transform(df['nursery'])
df['higher']= le.fit_transform(df['higher'])
df['internet']= le.fit_transform(df['internet'])
df['romantic']= le.fit_transform(df['romantic'])

#creating a new column final_grade in the dataframe from the mean of G1, G2 and G3
col = df.loc[: , "G1":"G3"]
df['final_grade'] = col.mean(axis=1)

#new csv file after adding final_grade column
df.to_csv('C:/Users/PRUSTY/Downloads/DS_ML_FinalTask/DS_ML_FinalTask/student-math-finally.csv', sep = ';')

#store final_grade column as an array in y
y = df['final_grade'].to_numpy()

#store all columns upto G2 in x as array
col = df.loc[:,"school":"G2"]
x = col.to_numpy()

#function to predict and test the accuracy of the predicted value with the true value
def predict(x):
    #splitting the dataset
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    
    #fitting linear regression to the train set
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    
    #predicting the output from testing set of x
    prediction = reg.predict(x_test)
    
    #calculate Accuracy Score
    from sklearn.metrics import r2_score
    print("Prediction Accuracy = ",r2_score(y_test, prediction)*100, "%")
    print("Test Accuracy = ",reg.score(x_test, y_test)*100, "%")    

    #visualisation of scatter plot between true value and predicted value
    plt.scatter(y_test, prediction, color = 'b')
    plt.xlabel('True Value --->')
    plt.ylabel('Predicted Value --->')
    plt.show()

#Creating backward elimination model for optimisation of the dataset
import statsmodels.api as smt
def bkwdelm(x,sl):
    k = len(x[0])
    for i in range(0,k):
        reg_OLS = smt.OLS(y,x).fit()
        Max = max(reg_OLS.pvalues).astype(float)
        if Max > sl:
           for j in range(0,k-i):
               if (reg_OLS.pvalues[j].astype(float) == Max):
                  x = np.delete(x,j,1)
    print(reg_OLS.summary())
    return x
    
sl = 0.005
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
x_modeled = bkwdelm(x_opt, sl)

#Calling the function "predict" to predict the result and calculate the accuracy with the optimised dataset
predict(x_modeled)