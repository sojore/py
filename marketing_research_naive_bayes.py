# -*- coding: utf-8 -*-
"""marketing_research_naive_bayes.ipynb

Automatically generated by Colaboratory.


#importing neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
df=pd.read_excel('/content/drive/MyDrive/marketin_ research_data.xlsx')
df

#dropping unnecesary columns
df1=df.drop('No',axis=1)
df1

#downscaling the age column for better analysis
df1['Age']=df1['Age']/20
df2=df1.copy()
df2

#converting the categorical data columns into numerical data columns
df3=df2.replace(['low','average','high'],[0,1,2])
df3.head()

df4=df3.replace(['no','yes'],[0,1])
df4.head()

df5=df4.replace(['good','very good'],[0,1])
df5

#obtaining the x_train and y_train(target) values
x_train=df5.drop('purchase of goods',axis='columns')
x_train.tail()

y_train=df5['purchase of goods']
y_train.sample(4)

##importing the naive bayes classifier algorithm libraries
from sklearn.naive_bayes import GaussianNB

#creating the model and training the model
model=GaussianNB()
model.fit(x_train,y_train)

#answering the question given problem
#(we need to tell if a person under the age of 25,with average income,graduate degree,good credit score is likely to purchase specific item)
##for that reason we need to note that a person under age of 25 is equivalent to lets say of age=24 which as per the data given above is equal
# to 1.2*20=24,so we will set age as =1.2
#average income is represented by =1
#graduate degree is represented by =1
#credit score is represented by =0
#we are now ready to go,

#we now call our model to predict if the above desribed individual is likely to purchase a comondity or not
#note that if we get a zero (0) value it means that the person is not likely to purchase comondity and 1 value implies the person is 
#likely to purchase a comondity
model.predict([[1.2,1,1,0]])

#there we go,so the above described individual will purchase the comondity and thus has a greater likelyhood to purchase the comondity

"""END OF DATACLEANING,MODEL CREATION,TRAINIG AND EVALUATION.THANK YOU!!!"""

