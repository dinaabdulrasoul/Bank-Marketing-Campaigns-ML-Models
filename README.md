# Optimizing an ML Pipeline in Azure
## Table of Contents
* ### Overview
* ### Summary
  * Problem Statement
  * Solution
* ### Pipelines
  * #### Scikit-learn Pipeline with Hyperdrive
     * Pipeline Architecture
     * Classification Algorithm
     * Parameter Sampler
     * Early Stopping Policy
  * #### AutoML Pipeline
* ### Pipelines Comparison
* ### Future Work

## Overview
This project is a part of the Udacity Azure ML Nanodegree.  
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.  
This model is then compared to an Azure AutoML run.  

## Summary
### **Problem Statement** 
The dataset contains data about marketing campaigns for a bank, the campaigns are based on phone calls. We seek to predict whether a bank product would be subscribed by the client or not (yes or no).  
The dataset contains 32950 training examples in a csv file.  
### **Solution**
We had two approaches to solving the problem, the first approach was by using Hyperdrive to obtain the best values of the hyperparamters for a scikit-learn logistic regression model, in order to maximize the accuracy of the model.  
The second approach, was to use Azure's automl to find the best performing model based on the highest accuracy value.  

## Scikit-learn Pipeline
* We first need to prepare our train.py script by:  
  * Importing the csv file containing the marketing campaigns data into our dataset using the TabularDataSetFactory module.  
  * Cleaning the dataset, which included droping NaN values.  
  * Splitting our dataset into training set (80% of the data) & test set (20% of the data.)   
  * Creating a Logistic Regression model using sci-kit learn.  
  * Creating a directory(outputs) to save the generated model into it.  
* After the train.py script is ready, we choose a proper parameter sampling method for the inverse regularization paramter(C) & the maximum number of iterations(max_iter), early termination policy and an estimator to create the HyperDriveConfig.  
  * The HyperDriveConfig was configured using the following:  
                             the estimator we created for the train.py,  
                             Paramater sampling method chosen,  
                             The early termination policy chosen,  
                             primary_metric_name, which is the Accuracy,  
                             primary_metric_goal, which is to maximize the primary metric,  
                             max_total_runs=4,  
                             max_concurrent_runs=4  
* Then we submit the hyperdrive run.  
* Once the run is complete, we choose the best run (the run that achieved the maximum accuracy) and save the model generated.  
 The best value of the Accuracy was found to be: **0.9072837632776934**  
 
The following diagram summarizes the workflow:  
![Scikit-learn Pipeline](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/hyperdrive_pipeline.PNG)  

**Algorithm**   
Logistic Regression is a supervisied binary classification algorithm that predicts the probability of a target varaible, returning either 1 or 0 (yes or no).  

**Parameter Sampler**  
For this pipeline, Random sampling has been used.  
Random Sampling is a great sampler to avoid bias, and it also supports early termination of low-performance runs.  

**Early Stopping Policy**  
For this pipeline, Bandit Policy has been used, which is an early termination policy based on slack criteria, and the evaluation interval.    
* Slack_factor is the ratio used to calculate the allowed distance from the best performing experiment run.  
* Evaluation_interval is the frequency for applying the policy.  
*The benefits of this stopping policy* is that any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated so this helps us quickly eliminate the bad performing runs.  

## AutoML  
**Best Performing Model**  
The best performing model found by the AutoML was the Voting Ensemble with an accuracy of 0.91779.  
Voting Ensemble is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble, that involves summing the predictions made by classification models.  
![Best Run](optimizing-an-ml-pipeline/Screenshots/doc4 - azureml studio.PNG)  

**Pipeline** 
 * Importing the csv file containing the marketing campaigns data into our dataset using the TabularDataSetFactory module.  
 * Importing "clean" function from train.py to clean the dataset, including droping NaN values.  
 * Splitting our dataset into training set (80% of the data) & test set (20% of the data.)
 * Preparing the AutoML Config by passing in the following:
    experiment_timeout_minutes=30  
    task="classification"  
    primary_metric="accuracy"  
    training_data=train_data  
    label_column_name="y"  
    n_cross_validations=2  
    max_concurrent_iterations=4  
    featurization='auto'  
  * Submitting the AutoML config.   
  * Finding the best run then saving the model.  

The following diagram summarizes the Pipeline workflow:  
![AutoML Pipeline](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/Automl_pipeline.PNG)  

Some of the learning algorithms that had been tested by the AutoML:
![Runs](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/Screenshots/doc1.PNG)  
![Runs](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/Screenshots/doc2.PNG)  
![Runs](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/Screenshots/doc3.PNG)  
![Runs](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/Screenshots/doc4.PNG)  


## Pipeline comparison  
The scikit-learn logisitc regression model, with the use of hyperdrive for hyperparameters tuning, achieved an accuracy of **0.9072837632776934**, while the automl voting ensemble model ahieved an accuracy of **0.91779**.  
Hyperdrive aims to optimize the paramaters of a specific learning algorithm while AutoML tries many different algorithms until it finds the model that obtains the highest value of Accuracy, this explains why AutoML achieved a higher accuracy; it tried diffferent algorithms not just one like hyperdrive. 
Pipeline-wise, AutoML didn't require a train.py script 


## Future work  
Some of the imporvements might be using a different sampling paramater, for example: BayesianSampling or GridSampling, they might take up more time and resources however that might improve the accuracy.  
Also we can try using a different termination policy or not using a termination policy at all, for example with BayesianSampling method as the data we have is not that large so an early termination policy is not really necessary.   

