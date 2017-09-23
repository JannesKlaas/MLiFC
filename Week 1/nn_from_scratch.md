# Building a neural net from scratch
This notebook aims to show students the inner workings of a basic neural network while at the same time iluminating some theoretical foundations that underpin machine learning in general. This work is largely inspired by [Trask's 'A neural network in 11 lines of Python'](https://iamtrask.github.io/2015/07/12/basic-python-network/).

## Maths
This exercise is quite mathematical and technical, but let me justify why I think knowing all this maths is important. Machine learning algorithms and neural networks in particular can seem a lot like magic. Because they are black boxes, that is we do not know how exactly they arrive at their results, the computations are too large for any human to follow, and the field has made such fast advances, many people have an unrealistic understanding of the technology. It seems to them like the computer is now genuinely intelligent and the rise of the robots is near. I have found that people who worked through the underlying math of a neural network gain a much more nuanced understanding and recognize its abilities and limitations much better.

## NN architecture
In this exercise, we will be working on a supervised learning problem. This means we got an input matrix $X$ and an output vector $y$ and we want to train our algorithm on making predictions for $y$. Let's give the data a look:

|$X_1$|$X_2$|$X_3$|$y$|
|-|-|-|---|
|0|1|0|0|
|1|0|0|1|
|1|1|1|1|
|0|1|1|0|

Our data has three input features, $X_1$,$X_2$ and $X_3$ and one output, $y$ with all values being either 1 or 0. You can also see that we have four examples in our data and that we combined the data into one matrix by stacking horizontal vectors. You might have noticed that $X_1$ is perfectly correlated with $y$. Finding those correlations automatically is exactly the task of our neural net. In this exercise we will implement a two layer network, also known as a perceptron.

### A perceptron

![Perceptron](/assets/Perceptron_moj.png)
TODO: Simplified image with sigmoid activation function.

As you can see in the image above, our neural net will consist of only one input layer and one output layer. The first the input features are multiplied with the weights $w$ and the weighted features then get summed up. This is exactly what happens in [linear regression](https://en.wikipedia.org/wiki/Linear_regression), a technique you might be familiar with. The outcome of our linear regression gets then passed into a non linear activation function, in our case the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). This is the same as what happens in [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression). 

### The sigmoid activation function

If we had a linear activation function, the output would simply be a weighted sum of our input features. Through non-linear activation functions we can model much more complex functions. In fact, it has been shown that neural nets can model _any_ function, as long as we make them big enough. Our sigmoid function has the formula:

$$S(X) = \frac{1}{1 + e^{-x}}$$
And looks like this:
![Sigmoid function](/assets/Logistic-curve.svg)

We can define the Python function:
```python
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
```

### Initializing weights

Now let's go backward through our perceptron. Before we can work with the weights, we have to initialize them.
```python
# initialize weights randomly with mean 0
w0 = 2*np.random.random((3,1)) - 1
```
This creates a 3 by 1 weight matrix, mapping our three input features to our one output. It is important to initialize the weights randomly and not just set them all to zero for the sake of symmetry breaking. If all the weights where the same, then they would never be different from each other, and our neural net could not model any complex functions.

### A full forward pass

With all the elements in place, we can now obtain predictions from our network with a forward pass.

1. We do a linear step:
$$ z_0 = Xw_0$$
2. We pass the linear product through the activation function
$$A_1 = \sigma(z_0)$$
```python
import numpy as np
#Seed the random function to ensure that we always get the same result
np.random.seed(1)
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
#set up w0
w0 = 2*np.random.random((3,1)) - 1

#define X
X = np.array([[0,1,0],
              [1,0,0],
              [1,1,1],
              [0,1,1]])

#define y
y = np.array([[0,1,1,0]])
#do the linear step
z0 = np.dot(X,w0)
#pass the linear step through the activation function
A1 = sigmoid(z0)
#see what we got
print(A1)
```
The output should look something like this:
```
[[ 0.60841366]
 [ 0.45860596]
 [ 0.3262757 ]
 [ 0.36375058]]
 ```
 
 ## Measuring losses
 
 That does not look like $y$ at all! That is just random numbers! We now have to modify the weights so that we arrive at better predictions.
 In order to arrive at better predictions, we first have to quantify how badly we did. In classification the metric used is the [cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy), sometimes also called logistic loss or log loss. It is calculated as follows:
 
 $$L(w) = -\frac{1}{N} \sum_{n=1}^N [y_n * \log \hat y_n  +  (1-y_n)*\log(1-\hat y_n)]$$
 
 Let's go through this step by step.
 
 1. $L(w)$ is the loss function given the weights $w$ that where used to obtain the prediction $\hat y_n$
 2. $-\frac{1}{N} \sum_{n=1}^N$ The loss over a batch of N examples is the average loss of all examples. We have factored a - out of the sum and moved it in front of the equation.
 3. $y_n * \log \hat y_n$ This part of the loss only comes into play if the true value, $y_n$ is 1. If $y_n$ is 1, we want $\hat y_n$ to be as close to 1 as possible, to achieve a low loss.
 4. $(1-y_n)*\log(1-\hat y_n)$ This part of the loss comes into play if $y_n$ is 0. If so, we want $\hat y_n$ to be close to 0 as well.
 
 ### Gradient descent
 
 TODO: IMAGE HERE
 
 Now that we have a good measure for our loss, how do we decrease it. This is a classic minimization task. We want to find a value for $w$ that minimizes $L(w)$. In linear regression we know how to solve the loss function for a value of $w$ that minimizes the squared loss function. For neural networks, this is sadly not possible. But what we can do instead is follow the gradient through a method called gradient descent.
 