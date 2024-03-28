#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libaray 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r"C:\Users\CHETAN\OneDrive\Desktop\winequality-red.csv")
df.head(10)


# In[3]:


df.tail(10)


# In[4]:


#consider 6 no. or higher is good quality and reaminaing is less than is not good 
Q = []
for i in df['quality']:
    if i >=6:
        i= 'good'
        Q.append(i)
    else:
        i='not good'
        Q.append(i)


# In[5]:


# added list data to dataframe 
df['Quality'] = Q
df


# In[6]:


# replaced quality data good is 1 and not good is 0
df['Quality'] = df['Quality'].replace({'good':1,'not good':0})
df


# In[7]:


# check label count and data is balanced 
df['Quality'].value_counts()


# In[8]:


# drop the quality data in default in data beacause we add new quality data  
df.drop('quality',axis=1,inplace=True)


# In[9]:


# checking information 
df.info()


# no null present in feature and not present object value 

# In[10]:


sns.heatmap(df.isnull())


# another proof is non null present in data using heatmap 

# In[11]:


for i in df.columns:
    x = df[i].value_counts()
    print(x)


# check value count method for all column and see no null , no white space value is present in data and check unique values

# In[12]:


df.describe()


# 1)fixed acidity is right skewness present and also present outlier
# 
# 2)volatile acidity and citric acid is normal bell curve distribution
# 
# 3)residual sugar is right skewness present and also diffrance 75% and max valve outlier present
# 
# 4)chlorides is right skewness present and aslo diffrance 75 % and max valve outler present 
# 
# 5)free sulfur dioxide is right skewness present and  diffrance 75% and max valve outlier present
# 
# 6)total sulfur dioxide is right skewness present and diffrance 75% and max valve oulter present
# 
# 7)sulphates is right skewness present and diffrance 75 % and max valve outlier present 
# 

# In[13]:


plt.figure(figsize=(30,25))
plotnumber = 1

for i in df:
    if plotnumber<=12:
        plt.subplot(3,4,plotnumber)
        sns.distplot(df[i])
        plt.xlabel(i,fontsize=20)
    plotnumber+=1 
plt.show()    


# check using distplot how data is distributed 
# 
# 1) volatile acidity, alcohol, citric acid is look like minimum right skewness and outlier 
# 
# 2) residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, sulphates is high right skewness and higher outler 
# 
# 3) fixed acidity, density, pH is look like normal distribution 

# In[14]:


# check skewness
df.skew()


# also check using df.skew method how data present skewness 
# we considerd  in range -0.5 and 0.5 not present skewness  

# In[15]:


# check outlier using boxplot
plt.figure(figsize=(30,25))
pltn = 1

for b in df:
    if pltn<=12:
        plt.subplot(3,4,pltn)
        sns.boxplot(df[b])
        plt.xlabel(b,fontsize=20)
    pltn+=1
plt.show()    


# outlier present in all feature except citric acid
# 
# drop outlier using zscore method 

# In[16]:


from scipy import stats 
from scipy.stats import zscore


# In[17]:


# we consider 60% data.
z_score = zscore(df)
abs_zscore = np.abs(z_score)
entery = (abs_zscore<2).all(axis=1)
df = df[entery]
df.describe()


# In[18]:


df.shape


# In[19]:


# after apply zscore we check skewness using df.skew method 
df.skew()


# In[20]:


# check also using distplot how data is distributed 
plt.figure(figsize=(30,25))
plotnumber = 1

for i in df:
    if plotnumber<=11:
        plt.subplot(3,4,plotnumber)
        sns.distplot(df[i])
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.show()    


# In[21]:


# seprate the feature and label 
x = df.drop('Quality',axis=1)
y = df['Quality']


# In[26]:


# check feature vs label realation 
plt.figure(figsize=(30,25))
plotnumber = 1

for i in df:
    if plotnumber<=12:
        plt.subplot(3,4,plotnumber)
        sns.stripplot(x=y,y=df[i],hue=y)
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.tight_layout()    


# In[27]:


plt.figure(figsize=(30,25))
plotnumber = 1

for i in df:
    if plotnumber<=12:
        plt.subplot(3,4,plotnumber)
        sns.barplot(x=y,y=df[i],hue=y)
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.tight_layout() 


# In[30]:


plt.figure(figsize=(30,25))
plotnumber = 1

for i in df:
    if plotnumber<=12:
        plt.subplot(3,4,plotnumber)
        sns.lineplot(x=y,y=df[i])
        plt.xlabel(i,fontsize=20)
    plotnumber+=1
plt.tight_layout() 


# In[31]:


x.shape


# In[32]:


y.value_counts()


# In[33]:


# check correlation with corrwith 
x.corrwith(y)


# In[34]:


# visulaize the correlation with label 
x.corrwith(y).plot(kind='bar',grid=True,figsize=(8,5),title='correlation with label')
plt.show()


# sulphates and alchol is strongly possitve reation with label
# 
# volatile acidity is strongly negative relation with label

# In[35]:


# visualze using heat map and check multicolineary problem 
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),linewidths=0.1,annot=True)
plt.yticks(rotation=0)
plt.show()


# residual sugar, free sulfur dioxide and ph feature is less contribute to label

# In[36]:


from sklearn.preprocessing import StandardScaler   
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[37]:


scaled = StandardScaler()
Scaled = pd.DataFrame(scaled.fit_transform(x),columns=x.columns)
Scaled


# In[38]:


# finding variance inflaction factor for each column
vif = []
for i in range(Scaled.shape[1]):
    X = variance_inflation_factor(Scaled,i)
    vif.append(X)


# In[39]:


VIF = pd.DataFrame()
VIF['vif'] = vif 
VIF['feature'] = x.columns
VIF


# consider vif >= 5 
# 
# fixed acidity and density have multicolinearity problem 
# 

# In[40]:


# consider fixed acidity feature to drop beacuse more vif value is present 
Scaled = Scaled.drop('fixed acidity',axis=1)


# In[41]:


v = []
for i in range(Scaled.shape[1]):
    V = variance_inflation_factor(Scaled,i)
    v.append(V)


# In[42]:


Vif = pd.DataFrame()
Vif['vif'] = v 
Vif['feature'] = Scaled.columns
Vif


# data have no multicolinearity problem 

# In[43]:


# check data is imblaced or not 
y.value_counts()


# In[44]:


sns.countplot(data=df,x='Quality')
plt.show()


# In[45]:


# split the data for train and test 
from sklearn.model_selection import train_test_split


# In[46]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=155)


# In[47]:


# build logitc regression model 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[48]:


LR = LogisticRegression()
LR.fit(x_train,y_train)


# In[49]:


y_pred = LR.predict(x_test)
# model accuracy 


# In[50]:


print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


# In[51]:


from sklearn.ensemble import GradientBoostingClassifier 


# In[52]:


gbdt = GradientBoostingClassifier()
gbdt.fit(x_train,y_train)


# In[53]:


y = gbdt.predict(x_test)
print(classification_report(y_test,y))
print(accuracy_score(y_test,y))


# In[54]:


from sklearn.neighbors import KNeighborsClassifier


# In[55]:


# initiate kneighbourclassifer

knn = KNeighborsClassifier()

# model training 
knn.fit(x_train,y_train)


# In[56]:


A = knn.predict(x_test)
print(classification_report(y_test,A))
print(accuracy_score(y_test,A))


# In[57]:


from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.tree import DecisionTreeClassifier


# In[58]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[59]:


DT =dt.predict(x_test)
print(accuracy_score(y_test,DT))
print(confusion_matrix(y_test,DT))
print(classification_report(y_test,DT))


# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)


# In[62]:


rfc = RFC.predict(x_test)
print(accuracy_score(y_test,rfc))
print(confusion_matrix(y_test,rfc))
print(classification_report(y_test,rfc))


# In[63]:


from sklearn.svm import SVC 


# In[64]:


svc = SVC()
svc.fit(x_train,y_train)


# In[65]:


SVC = svc.predict(x_test)
print(accuracy_score(y_test,SVC))
print(confusion_matrix(y_test,SVC))
print(classification_report(y_test,SVC))


# In[66]:


from sklearn.ensemble import AdaBoostClassifier


# In[67]:


ada = AdaBoostClassifier()
ada.fit(x_train,y_train)


# In[68]:


Ada = ada.predict(x_test)
print(accuracy_score(y_test,Ada))
print(confusion_matrix(y_test,Ada))
print(classification_report(y_test,Ada))


# In[69]:


from sklearn.ensemble import BaggingClassifier


# In[70]:


bag = BaggingClassifier()
bag.fit(x_train,y_train)


# In[71]:


Bag = bag.predict(x_test)
print(accuracy_score(y_test,Bag))
print(confusion_matrix(y_test,Bag))
print(classification_report(y_test,Bag))


# In[72]:


from sklearn.model_selection import cross_val_score


# In[73]:


score = cross_val_score(LR,x_train,y_train,cv=10)
print(score)
print(score.mean())
print(accuracy_score(y_test,y_pred) - score.mean())


# In[74]:


gbdt_score = cross_val_score(gbdt,x_train,y_train,cv=10)
print(gbdt_score)
print(gbdt_score.mean())
print(accuracy_score(y_test,y) - gbdt_score.mean())


# In[75]:


knn_score = cross_val_score(knn,x_train,y_train,cv=10)
print(knn_score)
print(knn_score.mean())
print(accuracy_score(y_test,A) - knn_score.mean())


# In[76]:


dt_score = cross_val_score(dt,x_train,y_train,cv=10)
print(dt_score)
print(dt_score.mean())
print(accuracy_score(y_test,DT) - dt_score.mean())


# In[77]:


RFC_score = cross_val_score(RFC,x_train,y_train,cv=10)
print(RFC_score)
print(RFC_score.mean())
print(accuracy_score(y_test,rfc) - RFC_score.mean())


# In[78]:


svc_score = cross_val_score(svc,x_train,y_train,cv=10)
print(svc_score)
print(svc_score.mean())
print(accuracy_score(y_test,SVC) - svc_score.mean())


# In[79]:


ada_score = cross_val_score(ada,x_train,y_train,cv=10)
print(ada_score)
print(ada_score.mean())
print(accuracy_score(y_test,Ada) - ada_score.mean())


# In[80]:


bag_score = cross_val_score(bag,x_train,y_train,cv=10)
print(bag_score)
print(bag_score.mean())
print(accuracy_score(y_test,Bag) - bag_score.mean())


# the best model is adaboostClassifier
# 
# apply hyperparameter tuning in adaboostClassifier model to increase accuracy 

# In[81]:


Ada = AdaBoostClassifier()

ada.fit(x_train,y_train)


# In[82]:


from sklearn.model_selection import GridSearchCV


# In[83]:


params = {'n_estimators':range(1,100,5),'learning_rate':np.arange(0.001,0.1,0.01)}


# In[84]:


grd_ada = GridSearchCV(AdaBoostClassifier(),param_grid=params)


# In[90]:


grd_ada.fit(x_train,y_train)


# In[91]:


grd_ada.best_estimator_


# In[92]:


Ada = AdaBoostClassifier(learning_rate=0.09099999999999998,n_estimators=91)

ada.fit(x_train,y_train)


# In[93]:


ADA = ada.predict(x_test)
print(accuracy_score(y_test,ADA))
print(confusion_matrix(y_test,ADA))
print(classification_report(y_test,ADA))


# plot ROc curve for AdaBoostClassifier 

# In[96]:


from sklearn import metrics


# In[97]:


fpr,tpr,threshold = metrics.roc_curve(y_test,ADA)
roc_auc = metrics.auc(fpr,tpr)
display = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc,estimator_name=Ada)
display.plot()


# AdaboostClassifier AUC score is 72%

# In[ ]:




