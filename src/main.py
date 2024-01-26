# From https://realpython.com/python-ai-neural-network/
# A neural network is a system that learns how to make predictions by following these steps:

# Taking the input data
# Making a prediction
# Comparing the prediction to the desired output
# Adjusting its internal state to predict correctly the next time

# Training a neural network is similar to the process of trial and error.
# Imagine you’re playing darts for the first time. In your first throw, you try to hit the central point of the dartboard.
# Usually, the first shot is just to get a sense of how the height and speed of your hand affect the result.
# If you see the dart is higher than the central point, then you adjust your hand to throw it a little lower, and so on.

# Working with neural networks consists of doing operations with vectors. You represent the vectors as multidimensional arrays.
# Vectors are useful in deep learning mainly because of one particular operation: the dot product.
# The dot product of two vectors tells you how similar they are in terms of direction and is scaled by the magnitude of the two vectors.

# The main vectors inside a neural network are the weights and bias vectors.
# Loosely, what you want your neural network to do is to check if an input is similar to other inputs it’s already seen.
# If the new input is similar to previously seen inputs, then the outputs will also be similar.
# That’s how you get the result of a prediction.

# Regression is used when you need to estimate the relationship between a dependent variable and two or more independent variables.
# Linear regression is a method applied when you approximate the relationship between the variables as linear.

# A linear relationship is one where there’s a direct relationship between an independent variable and a dependent variable.

# By modeling the relationship between the variables as linear,
# you can express the dependent variable as a weighted sum of the independent variables.
# So, each independent variable will be multiplied by a vector called weight.
# Besides the weights and the independent variables, you also add another vector: the bias.
# It sets the result when all the other independent variables are equal to zero.

# As a real-world example of how to build a linear regression model,
# imagine you want to train a model to predict the price of houses based on the area and how old the house is.
# You decide to model this relationship using linear regression.
# The following code block shows how you can write a linear regression model for the stated problem in pseudocode:

# price = (weights_area * area) + (weights_age * age) + bias

# In the above example, there are two weights: weights_area and weights_age.
# The training process consists of adjusting the weights and the bias so the model can predict the correct price value.
# To accomplish that, you’ll need to compute the prediction error and update the weights accordingly.

import numpy as np

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# Computing the dot product of input_vector and weights_1

# Vectors are objects which have both, magnitude and direction.
# Magnitude defines the size of the vector. It is represented by a line with an arrow,
# where the length of the line is the magnitude of the vector and the arrow shows the direction.
# For example, a line on a map showing the distance between two cities is a vector.
#
# Dot product is single value representation of multiple vectors. In this case how similar they are.
#
# We use numpy, but what it does behind the scenes:
#    first_indexes_mult = input_vector[0] * weights_1[0]
#    second_indexes_mult = input_vector[1] * weights_1[1]
#    dot_product_1 = first_indexes_mult + second_indexes_mult

dot_product_1 = np.dot(input_vector, weights_1)
dot_product_2 = np.dot(input_vector, weights_2)

print(f"The dot product is: {dot_product_1}")
print(f"The dot product is: {dot_product_2}")

# As a different way of thinking about the dot product, you can treat the similarity between the vector coordinates as an on-off switch.
# If the multiplication result is 0, then you’ll say that the coordinates are not similar.
# If the result is something other than 0, then you’ll say that they are similar.
#
# This way, you can view the dot product as a loose measurement of similarity between the vectors.
# Every time the multiplication result is 0, the final dot product will have a lower result.
# Getting back to the vectors of the example,
# since the dot product of input_vector and weights_2 is 4.1259,
# and 4.1259 is greater than 2.1672,
# it means that input_vector is more similar to weights_2.
