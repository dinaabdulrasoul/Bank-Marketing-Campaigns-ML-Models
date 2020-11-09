# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.



## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
The dataset contains data about marketing campaigns for a bank, the campaigns are based on phone calls. We seek to predict whether a bank product would be subscribed by the client or not (yes or no).
The dataset contained 32950 training examples in a csv file.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
We had two approaches to the problem, the first by using Hyperdrive to obtain the best hyperparamter, the model we used was a logistic regression model imported from sci-kit learn. 
In the second approach, automl was used to find the best performing model, which was...
## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
We first start by importing the data in the form of a csv file from a URL using the TabularDatasetFactory then we clean the data after that, 
the cleaned data is randomly split into train data (80% of the original dataset) & the test data (20% of the original dataset).
The classification algorithm used in this pipeline is the logistic regression algorithm.
Then we used hyperdrive to tune the inverse regularization paramter(regularization strength) & the maximum number of iterations in order to maximize the value of the accuracy. The best values were found to be: 'Regularization Strength:': 0.6174300437633223, 'Max iterations:': 100, 'Accuracy': 0.9072837632776934.
**What are the benefits of the parameter sampler you chose?**
Random sampling is a great sampler to avoid bias. It also supports early termination of low-performance runs.

**What are the benefits of the early stopping policy you chose?**
Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated so this helps us quickly eliminate the bad performing runs.
## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Some of the imporvements might be using a different sampling paramater, for example: BayesianSampling or GridSampling, they might take up more time and resources however that might improve the accuracy.
Also I can try not using a termination policy, for example with BayesianSampling method as the data we have is not that large so an early termination policy is not really necessary. 

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
