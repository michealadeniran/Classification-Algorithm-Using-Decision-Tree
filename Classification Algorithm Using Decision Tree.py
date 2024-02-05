#!/usr/bin/env python
# coding: utf-8

# # DECISION TREE ALGORITHM

# In[1]:


pip install -u imbalanced-learn


# ## IMPORTING LIBRARIES

# In[2]:


#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#loading dataset

Salary = pd.read_csv('Salary.csv')


# ## Inspecting our dataset

# In[4]:


Salary.shape


# In[5]:


#checking the first five row of the dataset
Salary.head()


# In[6]:


#checking the last five rows
Salary.tail()


# In[7]:


Salary.info()


# In[8]:


#checking for null Values


Salary.isnull().sum()


# In[9]:


#checking for duplicates

Salary.duplicated().value_counts()  


# In[10]:


#droppin the duplicates Values
Salary.drop_duplicates(keep='first',inplace=True)


# In[11]:


#duplicates values removes
Salary.shape


# In[12]:


Salary['workclass'].value_counts()


# In[13]:


#Replacing the ? with others
Salary['workclass'] = Salary['workclass'].str.replace('?', 'OthersWorkclass')


# In[14]:


Salary['workclass'].value_counts()


# In[15]:


Salary['fnlwgt'] .value_counts()


# In[16]:


Salary['education'] .value_counts()


# In[17]:


Salary['marital-status'].value_counts()  


# In[18]:


Salary['occupation'].value_counts()  


# In[19]:


Salary['occupation'] = Salary['occupation'].str.replace('?', 'UknownOccupation')


# In[20]:


Salary['occupation'].value_counts()  


# In[21]:


Salary['relationship'].value_counts()  


# In[22]:


Salary['race'].value_counts()


# In[23]:


Salary['sex'].value_counts()


# In[24]:


Salary['salary'].value_counts()


# In[25]:


Salary['native-country'].value_counts()


# In[26]:


Salary['native-country'] = Salary['native-country'].str.replace('?', 'OThersNativeCountry')


# In[27]:


#check if the changes have been effected
Salary['native-country'].value_counts()


# In[28]:


Salary.describe()


# In[29]:


Salary


# ## VISUALIZATION OF OUR DATASET

# In[30]:


#Plotting the workclass vs Hours-per-week
sns.barplot(x='workclass', y='hours-per-week', data=Salary)
plt.xlabel('workclass')
plt.tick_params(axis='x', rotation=90)
plt.ylabel('hours-per-week')
plt.title(' workclass vs. hours-per-week')
plt.show()


# In[31]:


sns.countplot(Salary, x="education" , hue='salary')
plt.tick_params(axis='x', rotation=90)


plt.show()


# In[32]:


sns.countplot(Salary, x="occupation" , hue='salary')
plt.tick_params(axis='x', rotation=90)


# In[34]:


sns.histplot(Salary.age)
plt.title('age')
plt.show()


# In[45]:


plt.figure(figsize=(8,4))
plt.title("Age Distribution of salary Earners above 50 or below")
sns.histplot(x="age", hue="salary", data=Salary)
plt.show()


# In[35]:


sns.countplot(Salary, x="native-country")
plt.tick_params(axis='x', rotation=90)


# In[36]:


sns.countplot(Salary, x="salary")


# In[37]:


sns.countplot(Salary, x="workclass" ,hue="salary" )

plt.tick_params(axis='x', rotation=90)


# In[46]:


Salary


# In[47]:


Salary['salary'].value_counts()


# ## SPLITTING OUR VARIABLE INTO OUTPUT AND INPUT

# In[57]:


Features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
            'occupation', 'relationship', 'race','capital-gain','capital-loss', 'sex',  'hours-per-week']


# In[58]:


X= Salary[Features].values

y= Salary['salary'].values


# In[69]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
sns.heatmap(Salary.corr(), annot=True, cmap='viridis')
plt.show()


# In[61]:


y


# In[62]:


from sklearn.preprocessing import OrdinalEncoder

# Create an instance of the OrdinalEncoder
encoder = OrdinalEncoder()

# Specify the columns you want to transform
categorical_columns = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

# Fit and transform the selected categorical columns
Salary[categorical_columns] = encoder.fit_transform(Salary[categorical_columns])

# Now, the specified categorical columns are transformed into numerical format


# In[66]:


Salary


# In[65]:


sns.countplot(Salary, x="salary")


# In[64]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

     


# In[67]:


from imblearn.over_sampling import RandomOverSampler
import seaborn as sns

# Create an instance of the RandomOverSampler
resampler = RandomOverSampler(random_state=0)

# Fit and transform the training data to oversample the minority class
X_train_oversampled, y_train_oversampled = resampler.fit_resample(X_train, y_train)

# Visualize the class distribution after oversampling
sns.countplot(x=y_train_oversampled)


# In[68]:


from sklearn.feature_selection import VarianceThreshold

# Create a VarianceThreshold instance with a threshold of 0
variance_selector = VarianceThreshold(threshold=0)

# Fit and transform the training data
X_train_fs = variance_selector.fit_transform(X_train)

# Transform the test data using the same selector
X_test_fs = variance_selector.transform(X_test)

# Print the number of features removed and the number of features that remain
print(f"{X_train.shape[1] - X_train_fs.shape[1]} features have been removed, {X_train_fs.shape[1]} features remain")


# ## SCALING 

# In[70]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_s=sc.fit_transform(X_train)
X_test_s=sc.transform(X_test)


# ## EVALUATING OUR MODEL

# In[72]:


#fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train_s, y_train)


# In[73]:


y_pred = classifier.predict(X_test_s)
print(y_pred)


# In[74]:


print(y_test)


# In[75]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[76]:


from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(cm,'\n\n')
print('-------------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[77]:


ax = sns.heatmap(cm, cmap='flare', annot=True, fmt='d')

plt.xlabel ("Predicted Class", fontsize =12)
plt.ylabel ("True Class", fontsize=12)
plt.title("Confusion Matix", fontsize=12)

plt.show()


# ## HYPERPARAMETER TUNING USING ADABOOST CLASSIFIER

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# adaBoost = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1),
# n_estimators=100)
# adaBoost.fit(X,y)
# y_pred = adaBoost.predict(X_test)

# In[80]:


y_pred


# In[83]:


from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(cm,'\n\n')
print('-------------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[ ]:




