import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df = pd.read_csv("/Users/sunilthapa/Desktop/My _projects/Linear_regression/Csv's/placement.csv")



# sns.scatterplot(data=df,x='cgpa',y='package')
# plt.show()


X = df.iloc[:,0:1]
y = df.iloc[:,-1]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()

lr.fit(X_train,y_train)

lr.predict(X_test.iloc[2].values.reshape(1,1))




## plotting regression line

# plt.scatter(df['cgpa'],df['package'])
# plt.plot(X_train, lr.predict(X_train), color = 'red')
# plt.show()



## y = mx + b     formula of linear regression

m = lr.coef_
b = lr.intercept_

## predicting the cgpa through formula x is cgpa and we are predicting y which is package


m * 9.8 + b
m * 5.8 + b


