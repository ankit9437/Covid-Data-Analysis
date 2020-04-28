# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:13:43 2020

@author: DELL
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import PolynomialFeatures


data=pd.read_csv("statewise.csv")

#print(data.head())
xasia=[]
for i in range(0,203,40):
    xasia.append(data.Date[i])

data["Date"]=pd.to_datetime(data['Date'])
data['Date']=data['Date'].map(dt.datetime.toordinal)
l=[]
li=[]
"""for i in data["State"]:
    if(i not in l):
        l.append(i)
for i in range(len(l)):
    li.append(i)
print(li)"""

l1=[]
stat=['Kerala', 'Tamil Nadu', 'Karnataka', 'Delhi', 'Andhra Pradesh', 'Madhya Pradesh', 'Odisha', 'Punjab', 'Chandigarh', 'Jammu and Kashmir', 'Maharashtra', 'Gujarat', 'Rajasthan']
val=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in range(13):
    l1.append(0)
data.State.replace(['Kerala', 'Tamil Nadu', 'Karnataka', 'Delhi', 'Andhra Pradesh', 'Madhya Pradesh', 'Odisha', 'Punjab', 'Chandigarh', 'Jammu and Kashmir', 'Maharashtra', 'Gujarat', 'Rajasthan'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
#print(l)
cases=[]
c=0
for i in range(len(data["Positive"])):
    cases.append(0)
for i in range(len(data["Positive"])):
    l1[data.State[i]]+=data.Positive[i]
    c+=data.Positive[i]
    cases[i]+=c


        
#da=data.drop("State",axis=1)
avg=data.Negative.mean()
std=data.Negative.std()
count=data.Negative.isnull().sum()
ran=np.random.randint(avg-std,avg+std,size=count)
data['Negative'][np.isnan(data['Negative'])]=ran



X=data.iloc[:,0:4]
Y=data.iloc[:,4]


#print(data.isnull().sum()/len(da)*100)
#print(Y)

#print(data['Date'])
#data["Date"]=pd.to_datetime(data["Date"],format='%dT-%m-%Y',errors='coerce')
#data["Date"]=pd.to_datetime(data.Date)
#print(data.dtypes)

#print(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)
classifier=LinearRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
yasia=[]
for i in range(0,203,40):
    yasia.append(cases[i])
    
print("Coefficients of features: ",classifier.coef_)
varience=classifier.score(x_test, y_test)
print(varience)
plt.scatter(classifier.predict(x_test), classifier.predict(x_test)-y_test, color='b', label="Test data")
plt.scatter(classifier.predict(x_train),classifier.predict(x_train)-y_train, color='r', label="Train")
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.title("Regression")
plt.legend(loc='upper right')
plt.show()

variance = classifier.score(x_test, y_test)
print(variance)
#print(len(data.Date))
#print(len(cases))"""
datesinp=[]
lii=[]
for i in range(len(data.Date)):
    lii=[]
    lii.append(data.Date[i])
    datesinp.append(lii)



    
xax=stat
yax=l1
plt.bar(xax, yax, tick_label = val, 
        width = 0.8) 
#plt.plot(xax,yax)
plt.xlabel=("States")
plt.ylabel("No of Cases")
plt.title("No of Cases in various States")
plt.show()


plt.plot(xasia,yasia)
plt.title("No of Cases in India with Time")
#plt.xlabel("Dates")
plt.ylabel("Total Cases")
plt.show()


model=LinearRegression()
model.fit(datesinp,cases)
dat=input("Enter the Date to be predicted")
dat=pd.to_datetime(dat)
at=dat.toordinal()
lii=[]
lii.append(at)
laa=[]
laa.append(lii)
pre=model.predict((laa))
pre=pre//1
print("answer is",*pre)






