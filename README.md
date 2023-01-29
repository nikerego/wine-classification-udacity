# Capstone Project - Azure Machine Learning Engineer

*TODO:* Write a short introduction to your project.
## Introduction
In this project, I train two machine learning models to perform a binary classification and compare their performance.
1. Automated ML (denoted as AutoML from now on) model: Trained using AutoML.
2. HyperDrive model: Logistic Regression with hyperparameters tuned using HyperDrive.

I demonstrate the ability to leverage an external dataset in my workspace, train the models using the 
different tools available in the AzureML framework as well as to deploy the best performing model as a web service.


## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. 
To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project
in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.


The Wine Quality datasets have been taken from the UCI Machine Learning Repository. The data is broken up into two 
individual datasets, one for red wines and the other for white wines. The red wine dataset contains 1599 examples while
the white wine dataset has 4898 examples. Both datasets have the same 12 attributes as follows:

Attribute information:

For more information, read [Cortez et al., 2009].

Input variables (based on physicochemical tests):
1. fixed acidity: numeric
2. volatile acidity: numeric
3. citric acid: numeric
4. residual sugar: numeric
5. chlorides: numeric
6. free sulfur dioxide: numeric
7. total sulfur dioxide: numeric
8. density: numeric
9. pH: numeric
10. sulphates: numeric
11. alcohol: numeric
12. quality: numeric

There are no missing values. I combine these datasets into one dataset and use the resulting dataset to for 
modeling in Azure. One model trained is a Logistic Regression whose hyperparameters are tuned by HyperDrive. The other 
model is trained is using AutoML. The goal for both models is to predict whether a wine is red or white. 

Response variable: (y)
13. y: (0: white, 1: red)

This dataset is public and available for research. I have included the citation below as requested:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
            [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
            [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib


### Access
*TODO*: Explain how you are accessing the data in your workspace.

The wine quality datasets live on the UCI Machine Learning Repository. I read them as a *pandas DataFrame* using the  
*read_csv* function and then store them on Azure 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

I set the following AutoML parameters: 

* *experiment_timeout_minutes*: 30
* *enable_early_stopping*: True    
* *primary_metric*: 'accuracy'
* *n_cross_validations*:5
* *iterations*: 50
* *max_concurrent_iterations*: 4

The *experiment_timeout_minutes* (30 mins), *iterations* (50) and *enable_early_stopping* (True) are set to reduce time 
taken for model training. Enabling early stopping allows the training process to conclude if there is no considerable 
improvement to the *primary_metric* (accuracy). *n_cross_validations* is set to 5 to ensure Bias vs Variance tradeoff
and prevent overfitting.The *max_concurrent_iterations* (4) is set as we are running on compute Standard_D2_V2 which has 
4 nodes. This allows 4 jobs to be run in parallel on each node.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters 
and their ranges used for the hyperparameter search

I used a Logistic Regression Classifier from Sklearn. My predictive task was binary classification and this model was 
appropriate. 

I chose to use tune 3 hyperparameters:
* *C*: Inverse of regularization strength; must be a positive float. Like in support vector machines, 
smaller values specify stronger regularization.
* *max_iter*: Maximum number of iterations taken for the solvers to converge.
* *solver*: Algorithm to use in the optimization problem.

Other possible hyperparameters were solver specific and were thus omitted. 
 
I used a RandomParameterSampling instead of an exhaustive GridSearch to reduce compute resources and time. This approach 
gives close to as good hyperparameters as a GridSearch with considerably less resources and time consumed. 

I used a BanditPolicy and set the evaluation_interval to 2 and the slack_factor to 0.1. This policy evaluates the primary 
metric every 2 iteration and if the value falls outside top 10% of the primary metric then the training process will stop. 
This saves time continuing to evaluate hyperparameters that don't show promise of improving our target metric. It prevents 
experiments from running for a long time and using up resources.

I chose the primary metric as Accuracy and to maximize it as part of the Hyperdrive run. I set the max total runs as 15 
to avoid log run times as well as the *max_concurrent_runs* to 4 as I am running this experiment on Standard_D2_V2 which 
has 4 nodes. This allows 4 jobs to be run in parallel on each node.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
