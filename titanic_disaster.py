import pandas as pd
import numpy as np

"""# READING DATA USING PANDAS"""

df = pd.DataFrame(pd.read_csv('/content/train (1).csv'))
df.head()

df.shape

"""**Description of the attribute of the dataset**

Pclass: Passenger Class(1 = 1st; 2= 2nd; 3 = 3rd)  
Survival: Survival(0 = No; 1 = Yes)  
name: Name  
sex: Sex  
age: Age  
SibSp: Number of Sibilings/Spouses Aboard  
parch:  Number of Parents/Children Aboard  
ticket: Ticket Number  
fare: Passenger Fare(POUND)  
cabin: Cabin  
embarked: Port Of Embarkation (C= Cherbourg; Q = Queenstown; S = Southampton)

# HANDELING NULL VALUES
"""

df.isnull().sum()

drop_col = df.isnull().sum()[df.isnull().sum()>(35/100 * df.shape[0])]
drop_col

drop_col.index

df.drop(drop_col.index, axis=1, inplace=True)
df.isnull().sum()

df.fillna(df.mean(), inplace=True)
df.isnull().sum()

"""Becuse **Embarked** contains string values, we see the details of the column seperately from others as strings does not have mean and all."""

df['Embarked'].describe()

"""For Embarked attribute, we fill the NULL values with most frequent value in the column"""

df['Embarked'].fillna('S', inplace =True)

df.isnull().sum()          ##NOW ALL THE NULL VALUES HAVE BEEN FILLED

df.corr()

"""SibSp: Number of Sibilings/Spouses Aboard  
Parch: Number of Parents/Children Aboard  

So we can make a new column damily_size by combaining these two columns
"""

df['FamilySize'] = df['SibSp']+df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
df.corr()

"""**FamilySize in the ship does not have much correlance with survival rate**   

Let's check if we weather the person was alone or not can affect the survival rate
"""

df['Alone'] = [0 if df['FamilySize'][i]>0 else 1 for i in df.index]
df.head()

df.groupby(['Alone'])['Survived'].mean()

"""If the person is alone he/she has less chance of surviving.  
                                                           
  The reason might be the person who is travleling with his family might be belonging to rich class and might be prioritized over other.
"""

df[['Alone', 'Fare']].corr()

"""So we can see if the person was not alone, the chance of the ticket price is higher"""

df['Sex'] = [0 if df['Sex'][i]=='male' else 1 for i in df.index]  # 1 for female, 0 for male
df.groupby(['Sex'])['Survived'].mean()

"""It shows, female passengers have been more chance of surviving than male ones.  
It shows women were prioritized over men
"""

df.groupby(['Embarked'])['Survived'].mean()

"""# **CONCLUSION**


*   Female passengers were prioritized over men.
*   People in high class or rich people have higher survival rate.
*   Passengers travelling with their family have higher survival rate.
*   Passengers who boarded the ship at Cherbourg survived more in proportion than the others.




"""
