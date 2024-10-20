#!/usr/bin/env python
# coding: utf-8

# **Logistical Regression by using a neural network**

# In[1]:


import math
import numpy as np
import pandas as pd

df = pd.read_csv('WineQT.csv' , sep = ',')
df.drop('Id', axis = 1, inplace = True)

def classify_quality(quality): #Making it binary for logistical regression. 0 = undrinkable 1 = drinkable
    return 1 if quality > 5 else 0

df['quality'] = df['quality'].apply(classify_quality)
df


# ![image.png](attachment:image.png)

# In[2]:


#Calculating z using the numpy.dot (it multiplies every feature with the corresponding weight)
def calc_z(weights,x,bias):
    return np.dot(weights, x) + bias


# In[3]:


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))  #Sigmoid-Function to only output values between 0 and 1


# ![image-4.png](attachment:image-4.png)

# In[4]:


#loss function between y and y_hat (Binary cross entropy)
def Bce(y_hat,y):
    # Clip y_hat to avoid log(0)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  #Prevent division by zero
    return - (y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))   


# ![image.png](attachment:image.png)
# 

# In[5]:


def derivative_x (y_hat,y,x):
    return x*(y_hat-y)
def derivative_b (y_hat,y):
    return y_hat-y


# ![image.png](attachment:image.png)

# In[6]:


def updateweights(weights,lr,derivative_x):
    return weights - lr* derivative_x
def updatebias(bias,lr,derivative_b): 
    return bias -lr* derivative_b


# In[7]:


#Initializing random weight, bias and learning rate 
weights = np.random.uniform(-1, 1, 11)  # Initialize weights randomly
bias = np.random.uniform(-1,1,1)
rows= 1142


# ![image.png](attachment:image.png)

# In[8]:


x = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
y = df['quality'].values


# In[9]:


def train_model(x, y,weights,bias,rows,epoch, lr,):
    print(weights)
    for epochs in range(epoch):
        total_loss = 0
        for i in range(rows):
            z = calc_z(weights,x[i],bias) 
            y_hat = sigmoid(z) #calculate expected y value
            loss = Bce(y_hat,y[i])
            total_loss = total_loss + loss
            deriv_x = derivative_x(y_hat, y[i], x[i]) 
            deriv_b = derivative_b(y_hat, y[i])
            weights = updateweights(weights,lr,deriv_x)
            bias = updatebias(bias,lr,deriv_b)
        if epochs % 10 == 0:
            print(f'Epoche {epochs}, Average Loss: {total_loss/rows}') 
    return weights,bias


# In[10]:


trained_weights, trained_bias = train_model(x,y,weights,bias,rows,300,0.001)


# In[11]:


correct_predictions = 0 
total_predictions = len(x)

for i in range(total_predictions):  # 
    z = calc_z(x[i], trained_weights, trained_bias)  
    prob = sigmoid(z)  # Calculate probability
    
    # Convert probability to binary output (0 or 1)
    prediction = 1 if prob > 0.5 else 0
    if prediction == y[i]:  # If the prediction matches the actual value
        correct_predictions += 1
        
accuracy = (correct_predictions / total_predictions) *100
print(f'The Accuracy that the neural network predicts the right outcome is: {accuracy}%')


# **References** \
# WhineQT.csv file : https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?resource=download \
# Numpy library : https://numpy.org \
# Pandas library: https://pandas.pydata.org \
# math library : https://docs.python.org/3/library/math.html \
# lecture files : (https://www.dropbox.com/scl/fo/7w9vvq7r6b7w0ffn7lfv9/AJlC3jzQeZIc5YuEmS5H8Ys?e=2&preview=From+Linear+Regression+to+Logistic+Regression.pdf&rlkey=5tx1qycellowj4e67nyu9ok5q&dl=0)
