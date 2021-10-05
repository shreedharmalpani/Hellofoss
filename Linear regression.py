#!/usr/bin/env python
# coding: utf-8

# ### Importing useful libraries 
# 

# In[135]:


# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in
import numpy as np   # linear algebra
import matplotlib.pyplot as plt # data visualization
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


# ### Loading the dataset 
# #### for implementation we will be using house prediction dataset . The dataset can be found [here](https://github.com/vrinda01go/Hellofoss/blob/main/house_price_data.txt)

# In[165]:


df = pd.read_csv("house_price_data.txt")  #import text file 
data=np.array(df)
plot_data(data[:,:2],data[:,-1])
normalize(data)


# #### feature normalization
# 

# In[150]:


def normalize(data1):
    #complete the code to normalize all data elements of each column 
    
    


# #### Plot data

# In[155]:


def plot_data(x,y):
    plt.xlabel('house size')
    plt.ylabel('price')
    plt.plot(x[:,0],y,'bo')
    plt.show()


# #### You can see that it is possible to roughly fit a line through the above plot. This means a linear approximation will actually allow us to make pretty accurate predictions and hence we go for linear regression.
# #### Well now that we have the data ready lets move on to the fun part. Coding the algorithm!

# #### our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis.
# #### for linear regression hypothesis is given by hθ(x) = θ0 + θ1x1 + θ2x2 + θ3x3 +…..+ θnxn.Since house price prediction dataset has two features i.e housesize and rooms ,our hypothesis function becomes
# ### hθ(x) = θ0 + θ1x1 + θ2x2 
# #### where x1 and x2 are the two features (i.e. size of house and number of rooms)

# ### Cost Function
# #### We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference (actually a fancier version of an average) of all the results of the hypothesis with inputs from x's and the actual output y's.
# 
# ![Screenshot%20%28345%29.png](attachment:Screenshot%20%28345%29.png)
# #### This function is otherwise called the "Squared error function", or "Mean squared error". Idea behind this is that we have to choose θ0,θ1,θ2 such that hθ(x) is close to y for our training example(x,y)
# 
# #### Your task is to define hypothesis and cost function which returns hypothesis function and cost function resp.

# In[ ]:


def h(x,theta):
    # complete this function 


# In[ ]:


def cost_function(x,y,theta):
    #complete this function


# ### Gradient Descent 
# #### So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.
# ### ![image.png](attachment:image.png)
# #### We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum.  The red arrows show the minimum points in the graph.
# 
# #### the way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter α, which is called the learning rate. 
# #### The gradient descent algorithm is:
# #### repeat until convergence  
# 
# 

# #### ![Screenshot%20%28347%29.png](attachment:Screenshot%20%28347%29.png)
# 
# #### We can substitute our actual cost function and our actual hypothesis function and modify the equation to :
# 

# ![Screenshot%20%28349%29.png](attachment:Screenshot%20%28349%29.png)

# #### Now since you know what gradient descent is , your next task is to define gradient descent function having learning rate =alpha and number of epochs =epochs

# In[ ]:


def gradient_descent(x,y,theta,alpha,epochs):
    
    #complete this function 


# #### For different values of theta we will get different values of our cost function .
# #### Now we want to visualize how our cost function varies with number of epochs .So your next task is to plot graph of updated costs vs number of epochs 

# #### After plotting above graph you will notice that your cost function decreases with epochs.
# #### Perfect! This is all what we wanted to seek by doing linear regression. 
# 
# #### Now it's time to test our model on some test data. 
# #### For this you will define a test function that will take as input the size of the house, the number of rooms and the final theta vector that was returned by our linear regression model and will give us the price of the house. Compute it for any value of x and fianl value of theta as given by gradient descent function

# In[ ]:


def test(theta,x):
    #complete this function.This function should return price of house i.e. y 


# #### Now since we have defined all required functions , we can call functions one by one and get our final results 

# In[ ]:


x,y = load_data("house_price_data.txt")
y = np.reshape(y, (46,1))
x = np.hstack((np.ones((x.shape[0],1)), x))
theta = np.zeros((x.shape[1], 1))
learning_rate = 0.1
num_epochs = 50
theta, J_all = gradient_descent(x, y, theta, learning_rate, num_epochs)
J = cost_function(x, y, theta)
print("Cost: ", J)
print("Parameters: ", theta)

#for testing and plotting cost 
n_epochs = []
jplot = []
count = 0
for i in J_all:
    jplot.append(i[0][0])
    n_epochs.append(count)
    count += 1
jplot = np.array(jplot)
n_epochs = np.array(n_epochs)
plot_cost(jplot, n_epochs)

test(theta, [1600, 2])

