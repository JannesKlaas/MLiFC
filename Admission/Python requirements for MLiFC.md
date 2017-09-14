# Python requirements for MLiFC

Welcome to MLiFC where you will learn about modern machine learning techniques in financial context. The course requires basic python as a perquisite. In this notebook you will find some exercises and tips. If you manage to do all the exercises, you should be prepared for the course. If you have trouble with these exercises, I recommend doing a python tutorial before you take this class. Some good ones are [Automate the boring stuff with Python](http://automatetheboringstuff.com/) or [codecademy's python tutorial](https://www.codecademy.com/learn/learn-python). If you have some experience with R, Matlab or some other language already it might be enough for you to just look up the specific syntax. I like the examples from [tutorialspoint](https://www.tutorialspoint.com/python3/index.htm) and the [official python documentation](https://docs.python.org/3/)

## Setup

The first exercise is to simply get this notebook running on your local machine. You can find an installation guide [here](http://jupyter.readthedocs.io/en/latest/install.html). I strongly recommend using Anaconda. My code is written in Python 3, to follow the exercises install anaconda 3.6

Once you have completed the install, you can download this notebook from GitHub. Start jupyter locally and navigate to the folder where you have stored this notebook. If you have trouble launching the notebook, refer to the [Jupyter beginner guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/) 

## Simple maths
To warm up, let's do some simple maths. 
Calculate: 

$$\frac{(24 + 5)^3}{5}$$

## Defining Functions
Now lets take the term above and turn it into a function.
Define the following as a python function: 

$$f(a,b,c,d) = \frac{(a + b)^c}{d}$$ 


## Loops
Loops play an important role in machine learning, as we have to go over data over and over again. Now, let's loop over the values 1 to 10 for a while setting  `b = 1; c = 2; d = 3` and print all results.

## Lists
Repeat the previous exercise but this time save all results to a [list](https://docs.python.org/2/tutorial/datastructures.html). Once you have the list, print the 3rd, 7th, and 8th item in the list.

## Numpy
Numpy is an important tool since it allows us to define and work with matrices. It is a library created for python. It should be included in the Anaconda installation, if you do not have it installed you can [install](https://anaconda.org/anaconda/numpy) it using `conda install -c anaconda numpy`
If you are unfamiliar with numpy, have a look at this [brief tutorial](http://cs231n.github.io/python-numpy-tutorial/#numpy) explaining the basics. If you have used Matlab before, you might find [this tutorial](http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users) helpful. If you need a broader introduction check out [data camp's tutorial](https://www.datacamp.com/community/tutorials/python-numpy-tutorial#gs.0d8n3ZI). 

Once you have numpy set up, create a vertical vector of four random numbers. You can use the `numpy.random.rand` function for this.

$$A=\begin{bmatrix}
         a \\
         b \\
         c\\
         d
        \end{bmatrix}
$$


### Element wise operations
Now create a second vector of the same dimensions. Perform an element wise addition and an element wise multiplication of the two vectors. Print the results.

$$B=\begin{bmatrix}
         e \\
         f \\
         g\\
         h
        \end{bmatrix}
$$
        
$$C=\begin{bmatrix}
         a + e \\
         b + f \\
         c + g\\
         d + h
        \end{bmatrix}
$$
        
$$D=\begin{bmatrix}
         a * e \\
         b * f \\
         c * g\\
         d * h
        \end{bmatrix}
$$

### Dot products
Calculate the outer and the inner product of the two vectors. You can use `numpy.dot` and the transpose operation for this. Print the results.

## Pandas
Finally we use Pandas to handle data. Again, it should be included in your anaconda installation. If it is not, follow the [installation instructions](https://pandas.pydata.org/pandas-docs/stable/install.html) While numpy does matrix operations, Pandas can handle files and table operations. I recommend you check out the [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html) tutorial to get an overview. If you like a longer, more beginner friendly explanation, check out [data camp's take](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python).

Go to the [kaggle page of the titanic dataset](https://www.kaggle.com/c/titanic). We will be working with the dataset later in the course. To download it you will need a kaggle account. It is free and I recommend you create one, as kaggle is a great platform to practice your ML skills after the course. For now, just load the `train.csv` dataset calculate some basic descriptive statistics using Pandas `describe()` function.

## Installing Keras
In class, we will use Keras with a Tensorflow backend as a deep learning library. You can install Keras via anaconda with `conda install -c conda-forge keras`. For more information see the [Keras documentation](https://keras.io/#installation). If you have installed Keras, make sure that it works by importing it here.

If your output looks like this, you are in business:


## Final words
If you managed to go through these introductory exercises and feel confident about it, you are ready for MLiFC. However, I strongly recommend you that you have a look at some other data science tools and maybe play around with them before starting the course. A nice tutorial on the titanic dataset we just worked with is on [kaggle](https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial). A nice introduction into some important helper libraries for machine learning can be found on the [CS231 Website](http://cs231n.github.io/python-numpy-tutorial/).