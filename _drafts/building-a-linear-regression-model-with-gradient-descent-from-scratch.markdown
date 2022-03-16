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

**The steps I followed to code my Logistic Regression model were:**

1. Code the sigmoid function
   * Takes X, theta
   * Returns the result of the sigmoid of X multiplied by theta
1. Code the cost function (a derivative of the sigmoid function)
   * Takes X, y, lambda (regularization coeff.)
   * Calls the sigmoid function
1. Code the gradient computation function (derivative of the cost)
   * Takes X, y, lambda
   * Returns the gradient
1. Code the gradient descent function to track the optimization of theta
   * Gradient descent takes X, y, lambda, alpha, number of iterations
   * Initializes theta as a 2D array
   * Generates a list of classes in y
   * For each class in y, for number of iterations:
      * Calls the cost and gradient functions
      * Multiplies alpha by the gradient
      * Subtracts the product from theta
      * Prints the cost after every 100 iterations
   * Ends inner loop when iterations is reached for each class
   * Assigns the computed class theta to the appropriate column in the initialized array theta
   * Ends outer loop when theta has been calculated and assigned for each class
   * Returns theta and the ordered list of classes in y

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



