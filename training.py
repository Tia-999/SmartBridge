#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
import warnings
import pickle
from scipy import stats
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[4]:


import pandas as pd
data = pd.read_csv("C:/Users/aswin/OneDrive/Desktop/thyroidDF.csv")


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# In[8]:


data.drop(["TSH_measured",'T3_measured','T4U_measured', "TT4_measured", "TBG_measured", "FTI_measured", "referral_source", "patient_id"], axis=1, inplace=True)


# In[9]:


diagnoses = {'A':"hyperthyroid conditions",
             'B':"hyperthyroid conditions",
             'C':"hyperthyroid conditions",
             'D':"hyperthyroid conditions",
             'E':"hypothyroid conditions",
             'F':"hypothyroid conditions",
             'G':"hypothyroid conditions",
             'H':"hypothyroid conditions",
             'I':"binding protein",
             'J':"binding protein",
             'K':"general health",
             'L':"replacement therapy",
             'M':"replacement therapy",
             'N':"replacement therapy",
             'O':"antithyroid treatment",
             'P':"antithyroid treatment",
             'Q':"antithyroid treatment",
             'R':"miscellaneous",
             'S':"miscellaneous",
             'T':"miscellaneous"}
data['target']=data['target'].map(diagnoses)


# In[10]:


data.dropna(subset=['target'],inplace=True)


# In[11]:


data['target'].value_counts()


# In[12]:


data.describe()


# In[13]:


data.info()


# In[14]:


data[data.age>100]


# In[15]:


data['age']=np.where((data.age>100),np.nan,data.age)


# In[16]:


data


# In[17]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[18]:


x


# In[19]:


y


# In[20]:


x['sex'].unique()


# In[21]:


x['sex'].replace(np.nan,'F',inplace=True)


# In[22]:


x['sex'].value_counts()


# In[23]:


x.info()


# In[24]:


x['age']=x['age'].astype('float')
x['TSH']=x['TSH'].astype('float')
x["T3"]=x["T3"].astype('float')
x["TT4"]=x['TT4'].astype('float')
x["T4U"]=x['T4U'].astype('float')
x["FTI"]=x['FTI'].astype('float')
x['TBG']=x['TBG'].astype('float')


# In[25]:


from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
ordinal_encoder= OrdinalEncoder(dtype='int64')
x.iloc[:,1:16]=ordinal_encoder.fit_transform(x.iloc[:,1:16])


# In[26]:


x


# In[27]:


data.dtypes


# In[28]:


x.replace(np.nan,'0',inplace=True)


# In[29]:


x


# In[30]:


label_encoder = LabelEncoder()
y_dt=label_encoder.fit_transform(y)


# In[31]:


y=pd.DataFrame(y_dt,columns=['target'])


# In[32]:


y


# In[33]:


import seaborn as sns
corrmat=x.corr()
f,ax=plt.subplots(figsize=(9,8))
sns.heatmap(corrmat,ax=ax,cmap="YlGnBu",linewidths=0.1)


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[37]:


from imblearn.over_sampling import SMOTE
y_train.value_counts()


# In[38]:


os= SMOTE(random_state=0,k_neighbors=1)
x_bal,y_bal=os.fit_resample(x_train,y_train)
x_test_bal,y_test_bal=os.fit_resample(x_test,y_test)


# In[39]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_bal=sc.fit_transform(x_bal)
x_test_bal=sc.transform(x_test_bal)


# In[40]:


x_bal


# In[41]:


columns=['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych','TSH','T3','TT4','T4U','FTI','TBG']


# In[42]:


x_test_bal=pd.DataFrame(x_test_bal,columns=columns)


# In[43]:


x_bal=pd.DataFrame(x_bal,columns=columns)


# In[44]:


x_bal


# In[45]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
rfr = RandomForestClassifier().fit(x_bal,y_bal)
y_pred = rfr.predict(x_test_bal)
accuracy_score(y_test_bal,y_pred)
x_bal.shape,y_bal.shape,x_test_bal.shape,y_test_bal.shape


# In[46]:


test_score=accuracy_score(y_test_bal,y_pred)


# In[47]:


test_score


# In[48]:


train_score=accuracy_score(y_bal,rfr.predict(x_bal))
train_score


# In[49]:


from sklearn.inspection import permutation_importance
results=permutation_importance(rfr,x_bal,y_bal,scoring='accuracy')


# In[50]:


feature_importance=['age','sex','on_thyroxine','quey_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych','TSH','T3','TT4','T4U','FTI','TBG']
importance = results.importances_mean
importance=np.sort(importance)
for i,v in enumerate(importance):
    i = feature_importance[i]
    print('feature:{:<20} Score: {}'.format(i,v))
plt.figure(figsize=(10,10))
plt.bar(x=feature_importance,height=importance)
plt.xticks(rotation=30,ha='right')
plt.show()


# In[51]:


x.head()


# In[52]:


x_bal


# In[53]:


x_bal= x_bal.drop(['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium'],axis=1)


# In[54]:


x_test_bal=x_test_bal.drop(['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_meds','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid','lithium'],axis=1)


# In[55]:


from sklearn.ensemble import RandomForestClassifier
rfr1 = RandomForestClassifier().fit(x_bal,y_bal.values.ravel())
y_pred = rfr1.predict(x_test_bal)
rfr1 = RandomForestClassifier()


# In[56]:


rfr1.fit(x_bal,y_bal.values.ravel())


# In[57]:


y_pred=rfr1.predict(x_test_bal)


# In[58]:


print(classification_report(y_test_bal,y_pred))


# In[59]:


train_score = accuracy_score(y_bal,rfr1.predict(x_bal))
train_score


# In[60]:


from xgboost import XGBClassifier
xgb1= XGBClassifier()
xgb1.fit(x_bal,y_bal)


# In[61]:


y_pred=xgb1.predict(x_test_bal)


# In[62]:


print(classification_report(y_test_bal,y_pred))


# In[63]:


accuracy_score(y_test_bal,y_pred)


# In[64]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
sv= SVC()


# In[65]:


sv.fit(x_bal,y_bal)


# In[66]:


y_pred=sv.predict(x_test_bal)


# In[67]:


print(classification_report(y_test_bal,y_pred))


# In[68]:


train_score=accuracy_score(y_bal,sv.predict(x_bal))
train_score


# In[69]:


params= {
    'C':[0.1,1,10,100,1000],
    'gamma':[1,0.1,0.01,0.001,0.0001],
    'kernel':['rbf','sqrt']
}


# In[70]:


from sklearn.model_selection import RandomizedSearchCV


# In[71]:


random_svc=RandomizedSearchCV(sv,params,scoring='accuracy',cv=5,n_jobs=-1)


# In[72]:


random_svc.fit(x_bal,y_bal)


# In[73]:


random_svc.best_params_


# In[74]:


svc1=SVC(kernel='rbf',gamma=0.1,C=100)


# In[75]:


svc1.fit(x_bal,y_bal)


# In[76]:


y_pred=svc1.predict(x_test_bal)


# In[77]:


print(classification_report(y_test_bal,y_pred))


# In[78]:


train_score=accuracy_score(y_bal,svc1.predict(x_bal))
train_score


# In[79]:


import pickle
filename = 'thyroid_model.pkl'
pickle.dump(new_sv,open(filename,'wb'))


# In[80]:


features=np.array([[0,0,0,0,0.000000,0.0,0.0,1.00,0.0,40.0]])
print(label_encoder.inverse_transform(xgb1.predict(features)))


# In[81]:


pickle.dump(label_encoder,open('label_encoder.pkl','wb'))


# In[82]:


data['target'].unique()


# In[83]:


y['target'].unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




