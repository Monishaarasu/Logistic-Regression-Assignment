#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries and dataset

# In[1]:


#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading the dataset
train=pd.read_csv('Titanic_train.csv')
train.head()


# In[3]:


train.shape


# In[4]:


test=pd.read_csv('Titanic_test.csv')
test.head()


# In[5]:


test.shape


# In[6]:


train.info()


# In[7]:


# Visualize the distribution of the 'Age' column
plt.figure(figsize=(8, 6))
sns.histplot(train['Age'], bins=20, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[8]:


# Visualize the survival rate based on gender
plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='Sex', hue='Survived')
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(["Did Not Survive", "Survived"])
plt.show()


# In[9]:


# Visualize the survival rate based on passenger class (Pclass)
plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='Pclass', hue='Survived')
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(["Did Not Survive", "Survived"])
plt.show()


# In[10]:


# Visualize the survival rate based on port of embarkation (Embarked)
plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='Embarked', hue='Survived')
plt.title("Survival Count by Port of Embarkation")
plt.xlabel("Port of Embarkation")
plt.ylabel("Count")
plt.legend(["Did Not Survive", "Survived"])
plt.show()


# In[11]:


sns.pairplot(train)
plt.show()


# In[12]:


#checking Missing Values
train.isnull().sum()


# In[13]:


train_df1=train.copy()


# In[14]:


train_df1.drop(['Cabin'],axis=1,inplace=True)


# In[15]:


train_df1.fillna({'Age':train_df1['Age'].mean(),
                        'Embarked':train_df1['Embarked'].mode()[0]},inplace=True)


# In[16]:


train_df1.isnull().sum()


# In[17]:


train_df1.drop(columns=['PassengerId','Name','Ticket'],inplace=True)


# In[18]:


train_df1.head(2)


# In[19]:


# Converting categorical variables to a dummy indicators
final_train= pd.get_dummies(train_df1, columns=['Sex'])
final_train= pd.get_dummies(final_train, columns=["Embarked"])


# In[20]:


final_train.head()


# In[21]:


final_train.shape


# # Loading Test dataset

# In[22]:


test.head()


# In[23]:


test_df=test.copy()


# In[24]:


test_df.drop(['Cabin'],axis=1,inplace=True)


# In[25]:


test_df.isnull().sum()


# In[26]:


test_df.fillna({'Age':test_df['Age'].mean()},inplace=True)


# In[27]:


test_df.fillna(0, inplace=True)


# In[28]:


test_df.isnull().sum()


# In[29]:


test_df.drop(columns=['Name','Ticket'],inplace=True)


# In[30]:


test_df.head(2)


# In[31]:


# Converting categorical variables to a dummy indicators
final_test= pd.get_dummies(test_df, columns=['Sex'])
final_test = pd.get_dummies(final_test, columns=["Embarked"])


# In[32]:


final_test.head(2)


# ## Train Test Split

# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X = final_train.drop('Survived',axis=1)
y = final_train['Survived']


# In[35]:


X.head(2)


# In[36]:


y.head(2)


# In[37]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ## Building the model

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# In[39]:


clf = LogisticRegression()


# In[40]:


clf.fit(X_train, y_train)


# In[41]:


train_predicted = clf.predict(X_train)


# In[42]:


cm = confusion_matrix(y_train, train_predicted)
cm


# In[43]:


print(classification_report(y_train, train_predicted))


# In[44]:


auc = roc_auc_score(y_train, train_predicted)
auc


# In[45]:


fpr, tpr, thresholds = roc_curve(y_train, clf.predict_proba (X_train)[:,1])

auc = roc_auc_score(y_train, train_predicted)

import matplotlib.pyplot as plt
x= 10
plt.plot(fpr, tpr, 
         color='blue'
        )

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[46]:


test_predicted = clf.predict(X_test)


# In[47]:


cm = confusion_matrix(y_test, test_predicted)
cm


# In[48]:


print(classification_report(y_test, test_predicted))


# In[49]:


auc = roc_auc_score(y_test, test_predicted)
auc


# In[50]:


fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba (X_test)[:,1])

auc = roc_auc_score(y_test, test_predicted)

import matplotlib.pyplot as plt
x= 10
plt.plot(fpr, tpr, 
         color='blue'
        )

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# ## Predictions

# In[51]:


test_x = final_test.drop('PassengerId',axis=1)


# In[52]:


predictions = clf.predict(test_x)


# In[53]:


final_prediction = pd.DataFrame({'PassengerId':final_test['PassengerId'],'Survived':predictions})


# In[54]:


final_prediction.head()


# Interview Questions:
#     
# 1. What is the difference between precision and recall?
# 
# Precision: The Measure of Exactness
# Precision is defined as the number of true positive predictions divided by the total number of positive predictions (true positives + false positives). It answers the question: Of all the instances that the model classified as positive, how many were actually positive?
# 
# Precision=TPTP+FP\text{Precision} = \frac{TP}{TP + FP}Precision=TP+FPTP​
# 
# True Positives (TP): Correctly predicted positive instances.
# False Positives (FP): Incorrectly predicted positive instances.
# 
# Recall: The Measure of Completeness
# Recall, also known as sensitivity or true positive rate, is the number of true positive predictions divided by the total number of actual positives (true positives + false negatives). It answers the question: Of all the actual positive instances, how many did the model correctly identify?
# 
# Recall=TPTP+FN\text{Recall} = \frac{TP}{TP + FN}Recall=TP+FNTP​
# 
# False Negatives (FN): Actual positive instances that were incorrectly predicted as negative.

# 2. What is cross-validation, and why is it important in binary classification?
# 
# Cross validation is a technique used in machine learning to evaluate the performance of a model on unseen data. It involves dividing the available data into multiple folds or subsets, using one of these folds as a validation set, and training the model on the remaining folds. This process is repeated multiple times, each time using a different fold as the validation set. Finally, the results from each validation step are averaged to produce a more robust estimate of the model’s performance. Cross validation is an important step in the machine learning process and helps to ensure that the model selected for deployment is robust and generalizes well to new data.
# 
# The main purpose of cross validation is to prevent overfitting, which occurs when a model is trained too well on the training data and performs poorly on new, unseen data. By evaluating the model on multiple validation sets, cross validation provides a more realistic estimate of the model’s generalization performance, i.e., its ability to perform well on new, unseen data.

# In[ ]:




