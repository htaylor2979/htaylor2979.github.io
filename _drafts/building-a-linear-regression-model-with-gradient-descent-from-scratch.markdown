---
title: Building a Linear Regression Model with Gradient Descent from Scratch
date: 2022-03-14 12:23:00 -04:00
categories:
- Machine Learning
tags:
- Python
---

## Introduction
The inspiration for this project came from Andrew Ng’s “Machine Learning” course on Coursera.org. The projects for that course used Octave math coding language, which has many similar functions to Python’s Numpy library. With that in mind, I decided to practice what I learned in Professor Ng’s course by building a Logistic Regression model from scratch in Python. I chose a simple gradient descent method for training the model weights and then tested my code with the famous “Iris” dataset that comes packaged with Python’s Scikit Learn library.

## Skills Demonstrated
* Python Libraries: Numpy, Seaborn, Pandas
* Logistic Regression coded with vectorized array functions
* Gradient Descent, also with vectorized array functions


## The Data
The first use of the Iris dataset is credited to Sir R.A. Fisher, and the data has frequently been used in demonstrating pattern recognition and classification in machine learning. For this project, I elected to use the version of the data that comes packaged with the Scikit Learn Python library. The target variable is the iris species, with three possible values. It is a small dataset with 150 rows and four features. The observations are divided evenly into 50 rows of each species, and there are no missing values.

### Data Features:
* Sepal Length (cm) 	
* Sepal Width (cm) 	
* Petal Length (cm) 	
* Petal Width (cm)

### Target Variable: Species
* setosa
* versicolor
* virginica

### Visualizing the Data
The scatter plots below, color coded by iris species, show the relationships among the variables. In particular, petal length and petal width appear to have a positive linear relationship across the three species. We also see that the setosa species (green dots) has distinctly different petal and sepal measurements from the other two species. So, at a glance, one would expect the model to perform consistently well on setosa. On the other hand, the measurements for versicolor and virginica (coral and blue dots) overlap in the plots. So, one might expect the model to be somewhat less accurate in classifying these two species.
<br />
<br />
<br />

*Figure 1: Pair plots comparing the relationships among data features*
![Pair Plots of Iris Features]({{ site.url }}/assets/img/iris-data/pairplot-iris-features.jpg)
<br />
<br />
<br />


### Python Code for Loading the Data and Creating the Pair Plots

{% highlight python %}
# Tables, arrays, and linear algebra
import numpy as np
import pandas as pd

# Datasets library
from sklearn import datasets

# Visualize the data
import seaborn as sns
sns.set_theme(style="ticks")
import matplotlib.pyplot as plt

# Load the data
iris = datasets.load_iris()

# Create a dataframe and set column labels
iris_df = pd.DataFrame(iris.data)
iris_df.columns=['Sepal Length',
  'Sepal Width',
  'Petal Length',
  'Petal Width']
iris_df['Species Code'] = iris.target

# Map iris species names to a new column
spec_names = iris.target_names
iris_df['Species Name'] = iris_df['Species Code'].apply(lambda x: spec_names[x])
{% endhighlight %}
<br />
<br />
<br />

*Figure 2: The first five rows of the Iris dataframe*
![Iris data first five rows]({{ site.url }}/assets/img/iris-data/iris-dataframe-head.jpg)
<br />
<br />
<br />


{% highlight python %}
# Context adjusts the font size and proportions of the graphics
sns.set_context("talk", font_scale=1.4)

# Create the plot
# Species Code is dropped since it is an integer code for the species name
plot = sns.pairplot(iris_df.drop('Species Code', axis=1), 
             hue="Species Name", 
             palette="Set2", 
             height=3.2,
            corner=True)

plt.show()
{% endhighlight %}
&nbsp;
&nbsp;

## The Model: Logistic Regression with Gradient Descent
Logistic regression uses the sigmoid function to model values between 0 and 1, which makes it useful for modeling True/False classifications. When choosing among multiple options, a different model is trained for each target value, and the results of each model are then compared to determine how best to classify each observation in the data.

The general form of the logistic regression model is a sigmoid function, which returns values between zero and one:
&nbsp;
&nbsp;

![Logistic regression sigmoid function]({{ site.url }}/assets/img/iris-data/sigmoid-function.jpg)
<br />
<br />
<br />

**The variables in the above equation are:**

* X = the inputs values (measurements of parts of the iris, in this case)
* y = the classification of each iris’s species
* θ = theta, representing an array of weights for each classification

It is worth noting that, while θ is referred to here as model “weights,” it functions similarly to coefficients used in algebra. Notation differs sometimes, but the general idea is that X represents multiple x-values of the data collected. The θ values are being optimized to produce the smallest amount of error when X values are input into the trained model. 

**The functions I coded for the Logistic Regression model are:**

1. Sigmoid function
   * Takes some value z
   * Returns the result of the sigmoid of z
1. Cost function (a derivative of the sigmoid function)
   * Takes X, y, lambda (regularization coeff.)
   * Returns the summation of cost for all rows of X
1. Gradient computation function (derivative of the cost)
   * Takes X, y, lambda
   * Returns the gradient
1. Gradient descent function to solve for theta
   * Takes an initial theta, X, y, lambda, alpha, number of iterations
   * Returns the calculated value of theta
1. Model training function for a binary y array
   * Takes X, y, lambda, number of iterations, alpha
   * Returns the new theta array received from gradient descent
1. Model training function for y with multiple classes
   * Takes X, y, lambda, number of iterations, alpha
   * Return the all thetas 2D array and the array of classes
1. Predict probabilities for all classes
   * Takes the all thetas array and X values for the predictions
   * Returns the result from the sigmoid function

Additional function used for prediction: *Numpy's argmax* function to find the column index for class that has the highest probability for each row 
   

<br />
<br />
<br />

### Python Code for the Training and Testing the Model


{% highlight python %}
# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z)) 
{% endhighlight %}


{% highlight python %}
# COST of Logistic Regression

# Inputs:
# X (without bias term) & y
# theta = weights, as a 2D array
# lam = regularization term lambda
# Returns cost (J)

def lr_cost(theta, X, y, lam):
    
    J = 0  # Initialize cost
    m = X.shape[0] # Number of examples/rows
    X = np.hstack((np.ones((1, m)).T, X))# Add ones for the bias theta
    
    # Check for case where a 1D array is passed in
    if len(theta.shape) == 1: 
        theta = theta.reshape((len(theta), 1))

    # First row of theta is for the bias term
    theta_reg = theta.copy()
    theta_reg[0] = 0  # Set bias term to zero
    
    tiny = .00000001 # a very small value
        
    # Regularization term
    reg_term = lam/(2 * m) * (theta_reg.T @ theta_reg)
    
    # Cost
    log_loss = 1/m * np.sum(
        (-1 * y * np.log(sigmoid(X @ theta) + tiny) 
         - (1 - y) * np.log(1 - sigmoid(X @ theta)+ tiny))) 
    
    J = log_loss + reg_term

    return J 
{% endhighlight %}

{% highlight python %}
# GRADIENT of Logistic Regression

# Inputs:
# X (without bias term) & y
# theta = weights, as a 2D array
# lam = regularization term lambda
# Returns gradient(grad)
def lr_gradient(theta, X, y, lam):
    
    # Check for cases where a 1D array is passed in
    if len(theta.shape) == 1: 
        theta = theta.reshape((len(theta), 1)).copy()
    
    if len(y.shape) == 1: 
        y = y.reshape((len(y), 1)).copy()
        
    m = X.shape[0] # Number of examples/rows
    grad = np.zeros(theta.shape)    # Initialize gradient   
    X = np.hstack((np.ones((1, m)).T, X))# Add ones for the bias theta
    
    # First row of theta is for the bias term
    theta_reg = theta.copy()
    theta_reg[0] = 0  # Set bias term to zero
    theta_reg = lam/m * theta_reg  # Gradient Regularization term 
    
    # keepdims option tells numpy sum not to flatten the axis of the result
    grad = np.sum((1/m * (sigmoid(X @ theta) - y) * X).T 
                  , axis=1
                  , keepdims=True) + theta_reg
    
    #print(grad)
    
    return grad
{% endhighlight %}

{% highlight python %}
# GRADIENT DESCENT
def gradient_desc(theta, X, y, lam, num_iters, alpha):
     
    m = X.shape[0] # Number of examples/rows

    # Check for 1D arrays (from using optimizers that flatten arrays)
    if len(y.shape) == 1: 
        y = y.reshape((len(y), 1)).copy()
    
    for i in range(num_iters):
        grad = lr_gradient(theta, X, y, lam)
        J = lr_cost(theta, X, y, lam)
        
        if i%100 == 0: print("Iteration: ", i, "Cost: ", J)
            
        # update theta
        theta = theta - alpha * grad
       
    return theta
{% endhighlight %}

{% highlight python %}
# TRAIN with Binary y

# Solve for Theta with simple binary y values
# lam = lamda in regularization terms
def lr_train(X, y, lam, num_iters, alpha):
    
    theta_size = X.shape[1] + 1
    init_theta = np.ones((theta_size, 1)) + .05

    theta = gradient_desc(init_theta, X, y, lam, num_iters, alpha)
    
    return theta
{% endhighlight %}

{% highlight python %}
# TRAIN with Multiple Class y

# Solves for Theta with multiple classes
# Returns
# - all_thetas: 2D array of thetas for ALL classes
# - class_array: array of class names, indexed same order as thetas
def lr_train_multi_class(X, y, lam=1, num_iters=1000, alpha=0.1):

    (m, n) = X.shape
    classes = np.unique(y)    
    
    # Initialize 2D array of thetas
    all_thetas = np.ones((n + 1, classes.shape[0]))
    
    i = 0
    for c in classes:
        
        print("Class: ", c)
        yc = np.array([1 if y == c else 0 for y in y])
    
        # Train the classifier on each class
        result = lr_train(X, yc, lam, num_iters, alpha)
        
        # Append predicted results as a new (row/col?) 
        all_thetas[0:, i] = result.flatten()

        i += 1

    return all_thetas, classes
{% endhighlight %}

{% highlight python %}
# PREDICT
# For multi class, each class's theta is a column vector
def lr_predict_prob_all(theta, X):
    
    (m, n) = X.shape
    X_ones_col = np.ones((1,m)).T
    
    # Check for case where a 1D array is passed in
    if len(theta.shape) == 1: 
        theta = theta.reshape((len(theta), 1))
    
    # Add a column of ones for the bias term in theta
    X = np.hstack((X_ones_col, X))
    
    # Predict all probabilities
    return sigmoid(X @ theta)
{% endhighlight %}



