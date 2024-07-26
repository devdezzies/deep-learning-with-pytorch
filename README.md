## DLP - Deep Learning with PyTorch
resources: 
- [PyTorch docs](https://pytorch.org/tutorials/)
- [learn-pytorch E-book](https://www.learnpytorch.io/)
- [PyTorch for Deep Learning & Machine Learning](https://www.youtube.com/watch?v=V_xro1bcAuA&t=6127s)

Chapter 0 – PyTorch Fundamentals
- Welcome and "what is deep learning?"
- Why use machine/deep learning?
- The number one rule of ML
- Machine learning vs deep learning
- Anatomy of neural networks
- Different learning paradigms
- What can deep learning be used for?
- What is/why PyTorch?
- What are tensors?
- Outline
- How to (and how not to) approach this course
- Important resources
- Getting setup
- Introduction to tensors
- Creating tensors
- Tensor datatypes
- Tensor attributes (information about tensors)
- Manipulating tensors
- Matrix multiplication
- Finding the min, max, mean & sum
- Reshaping, viewing and stacking
- Squeezing, unsqueezing and permuting
- Selecting data (indexing)
- PyTorch and NumPy
- Reproducibility
- Accessing a GPU
- Setting up device agnostic code

Chapter 1 – PyTorch Workflow
- Introduction to PyTorch Workflow
- Getting setup
- Creating a dataset with linear regression
- Creating training and test sets (the most important concept in ML)
- Creating our first PyTorch model
- Discussing important model building classes
- Checking out the internals of our model
- Making predictions with our model
- Training a model with PyTorch (intuition building)
- Setting up a loss function and optimizer
- PyTorch training loop intuition
- Running our training loop epoch by epoch
- Writing testing loop code
- Saving/loading a model
- Putting everything together

Chapter 2 – Neural Network Classification
- Introduction to machine learning classification
- Classification input and outputs
- Architecture of a classification neural network
- Turning our data into tensors
- Coding a neural network for classification data
- Using torch.nn.Sequential
- Loss, optimizer and evaluation functions for classification
- From model logits to prediction probabilities to prediction labels
- Train and test loops
- Discussing options to improve a model
- Creating a straight line dataset
- Evaluating our model's predictions
- The missing piece – non-linearity
- Putting it all together with a multiclass problem
- Troubleshooting a multi-class model

Chapter 3 – Computer Vision
- Introduction to computer vision
- Computer vision input and outputs
- What is a convolutional neural network?
- TorchVision
- Getting a computer vision dataset
- Mini-batches
- Creating DataLoaders
- Training and testing loops for batched data
- Running experiments on the GPU
- Creating a model with non-linear functions
- Creating a train/test loop
- Convolutional neural networks (overview)
- Coding a CNN
- Breaking down nn.Conv2d/nn.MaxPool2d
- Training our first CNN
- Making predictions on random test samples
- Plotting our best model predictions
- Evaluating model predictions with a confusion matrix

Chapter 4 – Custom Datasets
- Introduction to custom datasets
- Downloading a custom dataset of pizza, steak, and sushi images
- Becoming one with the data
- Turning images into tensors
- Creating image DataLoaders
- Creating a custom dataset class (overview)
- Writing a custom dataset class from scratch
- Turning custom datasets into DataLoaders
- Data augmentation
- Building a baseline model
- Getting a summary of our model with torchinfo
- Creating training and testing loop functions
- Plotting model 0 loss curves
- Overfitting and underfitting
- Plotting model 1 loss curves
- Plotting all the loss curves
- Predicting on custom data


