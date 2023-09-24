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

https://developers.google.com/machine-learning/crash-course/reducing-loss/video-lecture