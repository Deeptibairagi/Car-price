# Car-price
Creating for Car price prediction 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

df=pd.read_csv("car data.csv")
df.head()
df.shape

print(df["Seller_Type"].unique())
print(df["Transmission"].unique())
print(df["Fuel_Type"].unique())
print(df["Owner"].unique())

df.isnull().sum()
df.describe()
df.columns

final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()


final_dataset["Current year"]=2020
final_dataset.head()
final_dataset['no_years']=final_dataset['Current year']-final_dataset['Year']
final_dataset.head()

final_dataset.drop(['Year','Current year'],axis=1,inplace=True)
final_dataset.head()

final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.corr()

sns.pairplot(final_dataset)

corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')

final_dataset.head()

x=final_dataset.iloc[:,1:]     #independent variable
y=final_dataset.iloc[:,0]      #Dependent variable

x.head()
y.head()

model=ExtraTreesRegressor()
model.fit(x,y)

print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(5).plot(kind='barh')
plt.show()
