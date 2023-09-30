# Machine learning crash course

- [Machine learning crash course](#machine-learning-crash-course)
  - [Refs](#refs)
  - [Introduction to Machine Learning](#introduction-to-machine-learning)
    - [What is (supervised) machine learning? Concisely put, it is the following:](#what-is-supervised-machine-learning-concisely-put-it-is-the-following)
    - [Labels](#labels)
    - [Features](#features)
    - [Examples](#examples)
    - [Models](#models)
    - [Regression vs. classification](#regression-vs-classification)
    - [Some questions](#some-questions)
  - [Descending into ML](#descending-into-ml)
    - [Descending into ML: Linear Regression](#descending-into-ml-linear-regression)
    - [Descending into ML: Training and Loss](#descending-into-ml-training-and-loss)
      - [Squared loss: a popular loss function](#squared-loss-a-popular-loss-function)
  - [Reducing Loss](#reducing-loss)
    - [Reducing Loss: Gradient Descent](#reducing-loss-gradient-descent)
      - [Partial Derrivatives](#partial-derrivatives)
      - [Gradients](#gradients)
    - [Reducing Loss: Learning Rate](#reducing-loss-learning-rate)
      - [Maths aside on Learning Rate](#maths-aside-on-learning-rate)
    - [Reducing Loss: Stochastic Gradient Descent](#reducing-loss-stochastic-gradient-descent)
  - [First steps with TensorFlow](#first-steps-with-tensorflow)
    - [First Steps with TensorFlow: Programming Exercises](#first-steps-with-tensorflow-programming-exercises)
      - [NumPy and pandas](#numpy-and-pandas)
      - [Linear regression with tf.keras](#linear-regression-with-tfkeras)


## Refs

- notes from course https://developers.google.com/machine-learning/crash-course/ml-intro

## Introduction to Machine Learning

- can achieve many things difficult to quick do in logical programs
- using statistics rather than logic

## Key ML Terminology

### What is (supervised) machine learning? Concisely put, it is the following:

ML systems learn how to combine input to produce useful predictions on never-before-seen data.

### Labels

A label is the thing we're predicting—the y variable in simple linear regression. The label could be the future price of wheat, the kind of animal shown in a picture, the meaning of an audio clip, or just about anything.

### Features

A feature is an input variable—the x variable in simple linear regression. A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features, specified as:

In the spam detector example, the features could include the following:

- words in the email text
- sender's address
 -time of day the email was sent
 -email contains the phrase "one weird trick."

### Examples

An example is a particular instance of data, x. (We put x in boldface to indicate that it is a vector.) We break examples into two categories:

- labeled examples
 -unlabeled examples

A labeled example includes both feature(s) and the label. That is:

>  labeled examples: {features, label}: (x, y)

Use labeled examples to train the model. In our spam detector example, the labeled examples would be individual emails that users have explicitly marked as "spam" or "not spam."

For example, the following table shows 5 labeled examples from a data set containing information about housing prices in California:

| housing MedianAge (feature)	| totalRooms (feature)    | totalBedrooms (feature)	| medianHouseValue (label) |
| 15	                        | 5612                    | 	1283	            | 66900 |
| 19	                        | 7650                    | 	1901	            | 80100 |
| 17	                        | 720	                  |      174	            | 85700 |
| 14	                        | 1501                    | 	 337	            | 73400 |
| 20	                        | 1454                    | 	 326	            | 65500 |

An unlabeled example contains features but not the label. That is:

>  unlabeled examples: {features, ?}: (x, ?)

Here are 3 unlabeled examples from the same housing dataset, which exclude medianHouseValue:

| housingMedianAge  (feature)	| totalRooms (feature)	    | totalBedrooms (feature) |
| 42	                        | 1686	                    |  361 |
| 34	                        | 1226	                    |  180 |
| 33	                        | 1077	                    |  271 |

Once we've trained our model with labeled examples, we use that model to predict the label on unlabeled examples. In the spam detector, unlabeled examples are new emails that humans haven't yet labeled.

### Models

A model defines the relationship between features and label. For example, a spam detection model might associate certain features strongly with "spam". Let's highlight two phases of a model's life:

Training means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.

Inference means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions (y'). For example, during inference, you can predict medianHouseValue for new unlabeled examples.

### Regression vs. classification

A regression model predicts continuous values. For example, regression models make predictions that answer questions like the following:

What is the value of a house in California?

What is the probability that a user will click on this ad?

A classification model predicts discrete values. For example, classification models make predictions that answer questions like the following:

Is a given email message spam or not spam?

Is this an image of a dog, a cat, or a hamster?

### Some questions

![Supervised Learning](../images/supervised-learn.png)

![Features and Labels](../images/features-and-labels.png)

## Descending into ML

### Descending into ML: Linear Regression

Linear regression is a method for finding the straight line or hyperplane that best fits a set of points. 

It has long been known that crickets (an insect species) chirp more frequently on hotter days than on cooler days. For decades, professional and amateur scientists have cataloged data on chirps-per-minute and temperature. As a birthday gift, your Aunt Ruth gives you her cricket database and asks you to learn a model to predict this relationship. Using this data, you want to explore this relationship.

First, examine your data by plotting it:

![Cricket Chirps](../images/chirps.png)

**Figure 1. Chirps per Minute vs. Temperature in Celsius.**

As expected, the plot shows the temperature rising with the number of chirps. Is this relationship between chirps and temperature linear? Yes, you could draw a single straight line like the following to approximate this relationship:

![Cricket Chirps Line of Best Fit](../images/chirps-line-best-fit.png)

**Figure 2. A linear relationship.**

True, the line doesn't pass through every dot, but the line does clearly show the relationship between chirps and temperature. Using the equation for a line, you could write down this relationship as follows:

$y = mx + b$

where:

- $y$ is the temperature in Celsius—the value we're trying to predict.
- $m$ is the slope of the line.
- $x$ is the number of chirps per minute—the value of our input feature.
- $b$ is the y-intercept.

By convention in machine learning, you'll write the equation for a model slightly differently:

$y' = b + w_1x_1$

where:

- $y'$ is the predicted label (a desired output).
- $b$ is the bias (the y-intercept), sometimes referred to as 
- $w_1$ is the weight of feature 1. Weight is the same concept as the "slope" $m$ in the traditional equation of a line.
- $x_1$ is a feature (a known input).

To infer (predict) the temperature $y'$ for a new chirps-per-minute value $x_1$, just substitute the value into this model. Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight ($w_1$, $w_2$, etc.). For example, a model that relies on three features might look as follows:

$y' = b + w_1x_1 + w_2x_2 + w_3x_3$

### Descending into ML: Training and Loss

**Training** a model simply means learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

Loss is the penalty for a bad prediction. That is, **loss** is a number indicating how bad the model's prediction was on a single example. **If the model's prediction is perfect, the loss is zero**; otherwise, the loss is greater. **The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples**. For example, Figure 3 shows a high loss model on the left and a low loss model on the right. Note the following about the figure:

- The arrows represent loss.
- The blue lines represent predictions.

![Despiction of loss](../images/loss-depiction.png)

**Figure 3. High loss in the left model; low loss in the right model.**

 
Notice that the arrows in the left plot are much longer than their counterparts in the right plot. Clearly, the line in the right plot is a much better predictive model than the line in the left plot.

You might be wondering whether you could create a mathematical function—a loss function—that would aggregate the individual losses in a meaningful fashion.

#### Squared loss: a popular loss function

The linear regression models we'll examine here use a loss function called squared loss (also known as **$L_2$ loss**). The squared loss for a single example is as follows:

```
  = the square of the difference between the label and the prediction
  = (observation - prediction(x))2
  = (y - y')2
```

Mean square error (MSE) is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:

$MSE = \frac{1}{N} \sum_{(x,y)\in D} (y - prediction(x))^2$

where:

 - $(x,y)$ is an example in which
   - $x$ is the set of features (for example, chirps/minute, age, gender) that the model uses to make predictions.
   - $y$ is the example's label (for example, temperature).
 - $prediction(x)$ is a function of the weights and bias in combination with the set of features.
 - $D$  is a data set containing many labeled examples, which are $(x,y)$ pairs.
 - $N$ is the number of examples in $D$.

Although MSE is commonly-used in machine learning, it is neither the only practical loss function nor the best loss function for all circumstances.

## Reducing Loss

Here, in this module, you'll learn how a machine learning model iteratively reduces loss.

Iterative learning might remind you of the "Hot and Cold" kid's game for finding a hidden object like a thimble. In this game, the "hidden object" is the best possible model. You'll start with a wild guess ("The value of $w_1$ is 0.") and wait for the system to tell you what the loss is. Then, you'll try another guess ("The value of $w_1$ is 0.5.") and see what the loss is. Aah, you're getting warmer. Actually, if you play this game right, you'll usually be getting warmer. The real trick to the game is trying to find the best possible model as efficiently as possible.

The following figure suggests the iterative trial-and-error process that machine learning algorithms use to train a model:

![Loss Reduction Model](../images/loss-reduction-model.png)

**Figure 1. An iterative approach to training a model.**

We'll use this same iterative approach throughout the Machine Learning Crash Course, detailing various complications, particularly within that stormy cloud labeled "Model (Prediction Function)." Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.

The "model" takes one or more features as input and returns one prediction ($y'$) as output. To simplify, consider a model that takes one feature and returns one prediction:

$y' = b + w_1x_1$

What initial values should we set for $b$ and $w_1$? For linear regression problems, it turns out that the starting values aren't important. We could pick random values, but we'll just take the following trivial values instead:

- $b$ = 0
- $w_1$ = 0

Suppose that the first feature value is 10. Plugging that feature value into the prediction function yields:

$y' = 0 + 0 \cdot 10 = 0$

The "Compute Loss" part of the diagram is the loss function that the model will use. Suppose we use the squared loss function. The loss function takes in two input values:

- $y'$: The model's prediction for features x
- $y$: The correct label corresponding to features x.

At last, we've reached the "Compute parameter updates" part of the diagram. It is here that the machine learning system examines the value of the loss function and generates new values for $b$ and $w_1$. For now, just assume that this mysterious box devises new values and then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. And the learning continues iterating until the algorithm discovers the model parameters with the lowest possible loss. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has converged.

### Reducing Loss: Gradient Descent

The iterative approach diagram (Figure 1) contained a green hand-wavy box entitled "Compute parameter updates." We'll now replace that algorithmic fairy dust with something more substantial.

Suppose we had the time and the computing resources to calculate the loss for all possible values of $w_1$. For the kind of regression problems we've been examining, the resulting plot of loss vs. $w_1$ will always be convex. In other words, the plot will always be bowl-shaped, kind of like this:

![Bowl Shaped plot of loss](../images/bowl-shaped-plot.png)

**Figure 2. Regression problems yield convex loss vs. weight plots.**

Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.

Calculating the loss function for every conceivable value of $w_1$ over the entire data set would be an inefficient way of finding the convergence point. Let's examine a better mechanism—very popular in machine learning—called **gradient descent**.

The first stage in gradient descent is to pick a starting value (a starting point) for $w_1$. The starting point doesn't matter much; therefore, many algorithms simply set $w_1$ to 0 or pick a random value. The following figure shows that we've picked a starting point slightly greater than 0:

![Alt text](../images/starting-point.png)

**Figure 3. A starting point for gradient descent.**

The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. Here in Figure 3, the gradient of the loss is equal to the derivative (slope) of the curve, and tells you which way is "warmer" or "colder." When there are multiple weights, the **gradient** is a vector of partial derivatives with respect to the weights.

#### Partial Derrivatives

The math around machine learning is fascinating and we're delighted that you clicked the link to learn more. Please note, however, that TensorFlow handles all the gradient computations for you, so you don't actually have to understand the calculus provided here.

A **multivariable function** is a function with more than one argument, such as:

$f(x,y) = e^{2y}\sin(x)$

The **partial derivative** $f$ with respect to $x$, denoted as follows:

$\partial f \over \partial x$

is the derivative of $f$ considered as a function of $x$ alone. To find the following:

$\partial f \over \partial x$

you must hold $y$  constant (so $f$ is now a function of one variable $x$), and take the regular derivative of $f$ with respect to $x$. For example, when $y$ is fixed at 1, the preceding function becomes:

$f(x) = e^2\sin(x)$

This is just a function of one variable $x$, whose derivative is:

$e^2\cos(x)$

In general, thinking of $y$ as fixed, the partial derivative of $f$  with respect to $x$  is calculated as follows:

$\frac{\partial f}{\partial x}(x,y) = e^{2y}\cos(x)$

Similarly, if we hold $x$ fixed instead, the partial derivative of $f$ with respect to $y$ is:

$\frac{\partial f}{\partial y}(x,y) = 2e^{2y}\sin(x)$

Intuitively, a partial derivative tells you how much the function changes when you perturb one variable a bit. In the preceding example:

$\frac{\partial f}{\partial x} (0,1) = e^2 \approx 7.4$

So when you start at $(0, 1)$, hold $y$ constant, and move $x$ a little, $f$ changes by about 7.4 times the amount that you changed $x$.

In machine learning, partial derivatives are mostly used in conjunction with the gradient of a function.

#### Gradients

The gradient of a function, denoted as follows, is the vector of partial derivatives with respect to all of the independent variables:

$\nabla f$

For instance, if:

$f(x,y) = e^{2y}\sin(x)$

then:

$\nabla f(x,y) = \left(\frac{\partial f}{\partial x}(x,y), \frac{\partial f}{\partial y}(x,y)\right) = (e^{2y}\cos(x), 2e^{2y}\sin(x))$

Note the following:

- $\nabla f$ Points in the direction of greatest increase of the function.
- ${-\nabla f}$ Points in the direction of greatest decrease of the function.
- 
The number of dimensions in the vector is equal to the number of variables in the formula for $f$; in other words, the vector falls within the domain space of the function. For instance, the graph of the following function $f(x,y)$:

$f(x,y) = 4 + (x - 2)^2 + 2y^2$

when viewed in three dimensions with $z = f(x, y)$ looks like a valley with a minimum at $(2, 0, 4)$:

![three-d gully(../images/3d-gully.png)

The gradient of $f(x, y)$ is a two-dimensional vector that tells you in which $(x, y)$ direction to move for the maximum increase in height. Thus, the negative of the gradient moves you in the direction of maximum decrease in height. In other words, the negative of the gradient vector points into the valley.

In machine learning, gradients are used in gradient descent. We often have a loss function of many variables that we are trying to minimize, and we try to do this by following the negative of the gradient of the function.

Note that a gradient is a vector, so it has both of the following characteristics:

- a direction
- a magnitude

The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

![Negative Gradient](../images/negative-gradient.png)

**Figure 4. Gradient descent relies on negative gradients.**

To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point as shown in the following figure:

![Next Point](../images/next-point.png)

**Figure 5. A gradient step moves us to the next point on the loss curve.**

The gradient descent then repeats this process, edging ever closer to the minimum

### Reducing Loss: Learning Rate

As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the **learning rate** (also sometimes called **step size**) to determine the next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

**Hyperparameters** are the knobs that programmers tweak in machine learning algorithms. Most machine learning programmers spend a fair amount of time tuning the learning rate. If you pick a learning rate that is too small, learning will take too long:

![LR Too Small](../images/learning-too-small.png)

**Figure 6. Learning rate is too small.**

Conversely, if you specify a learning rate that is too large, the next point will perpetually bounce haphazardly across the bottom of the well like a quantum mechanics experiment gone horribly wrong:

![LR Too Large](../images/learning-too-large.png)

**Figure 7. Learning rate is too large.**

There's a Goldilocks learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.

![LR Perfect](../images/learning-perfect.png)

**Figure 8. Learning rate is just right.**

#### Maths aside on Learning Rate

The ideal learning rate in one-dimension is $\frac{ 1 }{ f''(x) }$ (the inverse of the second derivative of $f(x)$ at $x$).

The ideal learning rate for 2 or more dimensions is the inverse of the [Hessian](https://wikipedia.org/wiki/Hessian_matrix) (matrix of second partial derivatives).

The story for general convex functions is more complex.

### Reducing Loss: Stochastic Gradient Descent

In gradient descent, a **batch** is the set of examples you use to calculate the gradient in a single training iteration. So far, we've assumed that the batch has been the entire data set. When working at Google scale, data sets often contain billions or even hundreds of billions of examples. Furthermore, Google data sets often contain huge numbers of features. Consequently, a batch can be enormous. A very large batch may cause even a single iteration to take a very long time to compute.

A large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.

What if we could get the right gradient on average for much less computation? By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one. **Stochastic gradient descent (SGD)** takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.

**Mini-batch stochastic gradient descent (mini-batch SGD)** is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

To simplify the explanation, we focused on gradient descent for a single feature. Rest assured that gradient descent also works on feature sets that contain multiple features.


## First steps with TensorFlow

TensorFlow is an end-to-end open source platform for machine learning. TensorFlow is a rich system for managing all aspects of a machine learning system; however, this class focuses on using a particular TensorFlow API to develop and train machine learning models. See the [TensorFlow documentation](https://tensorflow.org/) for complete details on the broader TensorFlow system.

TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. Machine learning researchers use the low-level APIs to create and explore new machine learning algorithms. In this class, you will use a high-level API named tf.keras to define and train machine learning models and to make predictions. tf.keras is the TensorFlow variant of the open-source [Keras](https://keras.io/) API.

The following figure shows the hierarchy of TensorFlow toolkits:

![TensorFlow toolkits](../images/tensorflow-toolkits.png)

### First Steps with TensorFlow: Programming Exercises

As you progress through Machine Learning Crash Course, you'll put machine learning concepts into practice by coding models in tf.keras. You'll use Colab as a programming environment. Colab is Google's version of [Jupyter Notebook](https://jupyter.org/). Like Jupyter Notebook, Colab provides an interactive Python programming environment that combines text, code, graphics, and program output.

#### NumPy and pandas

Using `tf.keras` requires at least a little understanding of the following two open-source Python libraries:

- [NumPy](https://numpy.org/), which simplifies representing arrays and performing linear algebra operations.
- [pandas](https://pandas.pydata.org/), which provides an easy way to represent datasets in memory.

If you are unfamiliar with NumPy or pandas, please begin by doing the following two Colab exercises:

- [NumPy UltraQuick Tutorial Colab exercise- ](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=numpy_tf2-colab&hl=en), which provides all the NumPy information you need for this course.
- [pandas UltraQuick Tutorial Colab exercise](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=pandas_tf2-colab&hl=en), which provides all the pandas information you need for this course.

[Notes on NumPy](./NumPy-Quick-Tutorial.MD)

[Notes on Pandas](./Pandas-Quick-Tutorial.MD)

#### Linear regression with tf.keras

After gaining competency in NumPy and pandas, do the following two Colab exercises to explore linear regression and hyperparameter tuning in tf.keras:

- [Linear Regression with Synthetic Data Colab exercise](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_synthetic_tf2-colab&hl=en), which explores linear regression with a toy dataset. See notes [here](./Linear-Regression-With-Synthetic-Data.MD)
- [Linear Regression with a Real Dataset Colab exercise](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=linear_regression_real_tf2-colab&hl=en), which guides you through the kinds of analysis you should do on a real dataset. See notes [here](./Linear-Regression-With-Real-Data.MD)