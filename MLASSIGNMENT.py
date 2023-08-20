#!/usr/bin/env python
# coding: utf-8

# In[183]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import linalg


# In[184]:


class SVM_classifier():
    def __init__(self,learningrate,no_of_iteration,lamdaparameter):
        self.learningrate=learningrate
        self.no_of_iteration=no_of_iteration
        self.lamdaparameter=lamdaparameter
    def fit(self,X,Y):
        self.m,self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y
        for i in range(self.no_of_iteration):
            self.update_weights()
    def update_weights(self):
        y_label=np.where(self.Y<=0,-1,1)
        for index,x_i in enumerate(self.X):
            condition=y_label[index]*(np.dot(x_i,self.w)-self.b) >=1
            if(condition==True):
                dw=2*self.lamdaparameter*self.w
                db=0
            else:
                dw=2*self.lamdaparameter*self.w-(x_i*y_label[index])
                db=y_label[index]
            self.w=self.w-self.learningrate*dw
            self.b=self.b-self.learningrate*db
    def predict(self,X):
        output=np.dot(X,self.w)-self.b
        predicted_label=np.sign(output)
        y_hat=np.where(predicted_label<=-1,0,1)
        return y_hat


# In[185]:


data=np.loadtxt("train.dat")
print(data)
data.shape


# In[186]:


features=data[:,:-1]
features.shape


# In[187]:


target=data[:,-1:]
print(target)
target.shape


# In[188]:


def createfeatures(X):
    return np.cumprod(np.flip(-1+2*X,axis=1),axis=1)


# In[189]:


features=createfeatures(features)
print(features)


# In[190]:


features=np.c_[ features, np.ones(10000) ]   
features.shape
print(features)


# In[191]:


A=np.transpose(features)
B=linalg.khatri_rao(A,A)
C=linalg.khatri_rao(A,B)
D=np.transpose(C)
print(D)
D.shape


# In[192]:


Scaler=StandardScaler()
Scaler.fit(D)
standard_data=Scaler.transform(D)
D=standard_data
print(D)
target.shape


# In[193]:


clas=SVM_classifier(learningrate=0.001,no_of_iteration=10,lamdaparameter=0.01)
X_train,X_test,Y_train,Y_test=train_test_split(D,target,test_size=0.2)
clas.fit(X_train,Y_train)


# In[194]:


X_train_prediction=clas.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print(training_data_accuracy)


# In[195]:


X_test_prediction=clas.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print(test_data_accuracy)


# In[196]:


input_data=(0,1,0,0,1,0,1,1)
data1=np.asarray(input_data)
data2=data1.reshape(1,-1)
print(data2)


# In[197]:


data3=createfeatures(data2)
print(data3)


# In[198]:


data4=np.c_[ data3, np.ones(1) ] 
print(data4)


# In[199]:


A=np.transpose(data4)
B=linalg.khatri_rao(A,A)
C=linalg.khatri_rao(A,B)
D=np.transpose(C)
print(D)
D.shape


# In[200]:


std_data=Scaler.transform(D)
print(std_data)


# In[201]:


prediction=clas.predict(std_data)
print(prediction)


# In[ ]:




