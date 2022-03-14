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
The first use of the Iris dataset is credited to Sir R.A. Fisher, and the data has frequently been used in demonstrating pattern recognition and classification in machine learning. The target variable is the iris species, with three possible values. It is a small dataset with 150 rows and four features. The observations are divided evenly into 50 rows of each species, and there are no missing values.

#### Data Features:
* Sepal Length (cm) 	
* Sepal Width (cm) 	
* Petal Length (cm) 	
* Petal Width (cm)

#### Target Variable: Species
* setosa
* versicolor
* virginica

### Visualizing the Data
The scatter plots below give a quick comparison of 

*image_caption*
![Pair Plots of Iris Features]({{ site.url }}/assets/img/iris-data/pairplot-iris-features.jpg)
