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


*Figure 1: Pair plots comparing the relationships among data features*
![Pair Plots of Iris Features]({{ site.url }}/assets/img/iris-data/pairplot-iris-features.jpg)

## The Model: Logistic Regression with Gradient Descent
Logistic regression uses the sigmoid function to model values between 0 and 1, which makes it useful for modeling True/False classifications. When choosing among multiple options, a different model is trained for each target value, and the results of each model are then compared to determine how best to classify each observation in the data.

### Python Code for the Training and Testing the Model

*Variables:*

* X = the inputs values (measurements of parts of the iris, in this case)
* y = the classification of each iris’s species
* θ = theta, representing an array of weights for each classification

It is worth noting that, while θ is referred to here as model “weights,” it functions similarly to coefficients used in algebra. Notation differs sometimes, but the general idea is that X represents multiple x-values of the data collected. The θ values are being optimized to produce the smallest amount of error when X values are input into the trained model. 

The general form of the logistic regression model is a sigmoid function, which returns values between zero and one:
![Logistic regression sigmoid function]({{ site.url }}/assets/img/iris-data/sigmoid-function.jpg)

The steps I followed to code my Logistic Regression model were:
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
   * Initializes theta
   * Generates a list of classes in y
   * Loops through the y classes for number of iterations
      * Calls the cost and gradient functions
      * Multiplies alpha by the gradient
      * Subtracts the product from theta
      * Prints the cost after every 100 iterations
   * Ends loop when number of iterations is reached for all classes of y
   * Returns theta and the ordered list of classes in y






--- python
# Load a sample dataset
from sklearn import datasets

# Visualize the data
import seaborn as sns
sns.set_theme(style="ticks")
import matplotlib.pyplot as plt

iris = datasets.load_iris()

---

