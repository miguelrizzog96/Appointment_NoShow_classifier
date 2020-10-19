# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:23:47 2020

@author: Miguel Angel Rizzo Gonzalez
"""
#Importing libraries
import pandas as pd , matplotlib.pyplot as plt, seaborn as sns,numpy as np

#establishing file path
fpath='E:/Libros/py4e/Appointment no shows/KaggleV2-May-2016.csv'

#read the dataset
data=pd.read_csv(fpath)

# show the first 5 rows of the data
data.head()
#show data types
data.info()

# renaming columns to lowercase letters
data.columns=data.columns.map(lambda x: x.lower())
data.columns
#parsing dates
data['scheduledday']=pd.to_datetime(data['scheduledday'],utc=True)
data['appointmentday']=pd.to_datetime(data['appointmentday'],utc=True)
data[['scheduledday','appointmentday']].dtypes

#extracting day of the week from the appointment day
data['dow_appointment'] = data['appointmentday'].dt.weekday
data['dow_appointment'].value_counts()
#extracting day of the week from the scheduled day
data['dow_schedule'] = data['scheduledday'].dt.weekday
data['dow_schedule'].value_counts()
#extracting month of the appointment day
data['appointment_month']=data['appointmentday'].dt.month
data.appointment_month.value_counts()
#extracting month of the scheduled day
data['schedule_month']=data['scheduledday'].dt.month
data.schedule_month.value_counts()

#renaming column for convinience
data=data.rename(columns={'no-show': 'no_show'})


        
#retrive info about variables and vartypes
data.info()

#dictionaries for mapping variables to categorical
data.gender.value_counts()
rdict={'No':0,'Yes':1,'M':0,'F':1}
data.handcap.value_counts()
rdict2={4:1,3:1,2:1}
ndict={}

#renaming neighbourhoods with less than 200 patients with 'Other'
cn=data.neighbourhood.value_counts()
for i in cn.index:
    if cn[i] < 200:
        ndict.setdefault(i,'Other')

#replacing data with new values
data["no_show"] = data['no_show'].replace(rdict)
data["gender"] = data['gender'].replace(rdict)
data["handcap"] = data['handcap'].replace(rdict2)
data["neighbourhood"] = data['neighbourhood'].replace(ndict)

#computing time differential between day of schedule and appointment day
data['tdelta']=abs(data['appointmentday']-data['scheduledday'])
data.tdelta[0]
#convert from timedelta to numeric
data['tdelta']=pd.to_numeric(data['tdelta'])
data.tdelta[0]
#convert from nanoseconds to hours
data['tdelta']=pd.to_numeric(data['tdelta'])/(3600*1000000000)
data.tdelta[0]

#converting to categorical
data['categorical_tdelta'] = pd.qcut(data['tdelta'], 4)
data.categorical_tdelta.value_counts() 
#converting to categorical
data['categorical_age'] = pd.qcut(data['age'], 4)
data.categorical_age.value_counts()


#Showing Appointment no-shows relationship with independent variables

#Age
data[['categorical_age', 'no_show']].groupby(['categorical_age'], as_index=False).mean()
sns.barplot(x='categorical_age',y='no_show',data=data)
plt.show()
#Scholarship
data[['no_show', 'scholarship']].groupby(['scholarship'], as_index=False).mean()
sns.barplot(x='scholarship',y='no_show',data=data)
plt.show()
#Hipertension
data[['no_show', 'hipertension']].groupby(['hipertension'], as_index=False).mean()
sns.barplot(x='hipertension',y='no_show',data=data)
plt.show()
#Diabetes
data[['no_show', 'diabetes']].groupby(['diabetes'], as_index=False).mean()
sns.barplot(x='diabetes',y='no_show',data=data)
plt.show()
#Alcoholism
data[['no_show', 'alcoholism','gender']].groupby(['alcoholism','gender'], as_index=False).mean()
sns.barplot(x='alcoholism',y='no_show',hue='gender',data=data)
plt.show()
#Handicap
data[['no_show', 'handcap']].groupby(['handcap'], as_index=False).mean()
sns.barplot(x='handcap',y='no_show',data=data)
plt.show()
#Recieved SMS
data[['no_show', 'sms_received']].groupby(['sms_received'], as_index=False).mean()
sns.barplot(x='sms_received',y='no_show',data=data)
plt.show()
#Time Differential
data[['no_show', 'categorical_tdelta']].groupby(['categorical_tdelta'], as_index=False).mean()
sns.barplot(y='categorical_tdelta',x='no_show',data=data)
plt.show()
#Day of week of appointment
data[['no_show', 'dow_appointment']].groupby(['dow_appointment'], as_index=False).mean()

#Day of week of schedule
data[['no_show', 'dow_schedule']].groupby(['dow_schedule'], as_index=False).mean()
sns.barplot(x='dow_schedule',y='no_show',data=data,ci=None)
plt.show()
#Neighbourhood
data[['no_show', 'neighbourhood']].groupby(['neighbourhood'], as_index=False).mean()
plt.figure(figsize=(10,20))
sns.barplot(y='neighbourhood',x='no_show',data=data)
plt.show()

 # Mapping Age
data.loc[data['age'] <= 18, 'age'] = 0
data.loc[(data['age'] > 18) & (data['age'] <= 37), 'age'] = 1
data.loc[(data['age'] > 37) & (data['age'] <= 55), 'age'] = 2
data.loc[ data['age'] > 55, 'age'] =3  

 # Mapping time Delta
data.loc[data['tdelta'] <= 11.585, 'tdelta'] = 0
data.loc[(data['tdelta'] > 11.585) & (data['tdelta'] <= 83.38), 'tdelta'] = 1
data.loc[(data['tdelta'] > 83.38) & (data['tdelta'] <= 343.693), 'tdelta'] = 2
data.loc[ data['tdelta'] > 343.693, 'tdelta'] =3  

from sklearn.preprocessing import LabelEncoder
#Label encoding for independent variables
le_neighbourhood=LabelEncoder()
data['le_neighbourhood']=le_neighbourhood.fit_transform(data['neighbourhood'])
data.le_neighbourhood.value_counts()

#Selecting columns to drop for modeling
drop_elements = ['patientid','appointmentid','scheduledday','appointmentday','neighbourhood',
                 'categorical_age','no_show','categorical_tdelta']
#dropping columns

x=data.drop(drop_elements, axis = 'columns')
x.columns
x
y=data.no_show
y

#plotting value counts for each feature in the model
for i in x:
    plt.figure(figsize=(20,10))
    sns.countplot(x[i])
    plt.show()

# split into train-test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,stratify=y)

print(x.shape,y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Implementing Desicion tree classifier
from sklearn import tree
from sklearn.metrics import accuracy_score

# fit the model

model= tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

model.score(X_train,y_train)

y_pred=model.predict(X_test)

# evaluate predictions
acc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc)

pd.Series(y_pred).value_counts()

y_test.value_counts()

y_train.value_counts()

#Feature Importance
print("Feature Importance:\n")
for name, importance in zip(x.columns, np.sort(model.feature_importances_)[::-1]):
    print("{} -- {:.2f}".format(name, importance))
    
    
    
    
#Contingency table for predicted values and actual values

from sklearn.metrics import confusion_matrix

results = confusion_matrix(y_pred, y_test)
print(results)




