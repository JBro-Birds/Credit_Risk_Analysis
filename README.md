# Credit_Risk_Analysis

## Overview of Project
In 2019 there was a record-breaking number of more than 19 million Americans with at least one unsecured personal loan. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.  At an initial level of understanding machine learning models Jill, my data scientist boss, has asked me to research the difference between regression and classification models to predict credit risk.  Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.    

For this analysis I'll use Python and Scikit-learn, a Python machine learning library, to build and evaluate several machine learning models to predict credit risk.  The first model I'll use is logistic regression, a popular classification model.  With logistic regression, it is possible to try to answer questions such as whether a credit card holder is likely to miss a payment in the next month.  I'll follow the familiar pattern as I instantiate a model, train it, create predictions, then validate the model.  The model must answer important questions: How well does it perform? What percentage of predictions does it get right?  These questions will be answered by performing accuracy scoring and metrics covering sensitivity and precision.  Next, to improve the model's performance Jill has asked me to perform ensemble learning, random forest modeling and resampling techniques.

### Purpose
Jill has requested I perform a deep dive into the resampling aspect of machine learning to further enhance credit risk modeling.  Due to credit risk being an inherently unbalanced classification problem, as good loans easily outnumber risky loans, Jill has aksed me to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.  Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once completed, I’ll evaluate the performance of these models and write a recommendation on whether they should be used to predict credit risk.

## Linear Regression to Predict MPG
The dataset provided contains 50 vehicle prototypes with 6 variables (vehicle length, vehicle weight, spoiler angle, ground clearance, all-wheel-drive assignment, miles per gallon).

The linear regression is as follows:

![line_regression](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/line_regression.png)

The p-value and r-squared value for the linear regression is as follows:

![p_value_r_squared](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/p_value_r_squared.png)

*  Which variables/coefficients provided a non-random amount of variance to the mpg values in the dataset?  Based on the assumption that .05 level of significance is statistically significant the vehicle lenght (2.60e-12 coefficient) and ground clearance (5.21e-08 coefficient) variables provide a non-random amount of variance to the mpg values in the dataset.  The other three variables have coefficients > 0.05.

* Is the slope of the linear model considered to be zero? Why or why not?  With the p-value of 5.35e-11 being < 0.05 the slope is non-zero.

* Does this linear model predict mpg of MechaCar prototypes effectively? Why or why not?  The linear model predicts mpg of MechaCar prototypes effectively since the multiple R-squared value is 0.71.  Based on the Pearson correlation coefficient reference table r >= 0.7 signals a "strong" correlation.

## Summary Statistics on Suspension Coils
The dataset provided contains 150 vehicle IDs with corresponding PSI; the vehicle IDs are divided among 3 lots.

![total_summary](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/total_summary.png)

![lot_summary](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/lot_summary.png)

* The design specifications for the MechaCar suspension coils dictate that the variance of the suspension coils must not exceed 100 pounds per square inch. Does the current manufacturing data meet this design specification for all manufacturing lots in total and each lot individually? Why or why not?  The total for all manufacturing lots meets the specification with a variance of 62.3, well under the specified 100 pounds per square ince.  Analyzing the three individual lots tells a diferent story.  Lots 1 (0.98 variance) and 2 (7.47 variance) meet the specification while lot 3 (170.29 variance) is well above the specified value.

## T-Tests on Suspension Coils
T-test for all:
![t_test_lotAll](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/t_test_lotAll.png)

T-test for Lot 1:
![t_test_lot1](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/t_test_lot1.png)

T-test for Lot 2:
![t_test_lot2](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/t_test_lot2.png)

T-test for Lot 3:
![t_test_lot3](https://raw.githubusercontent.com/JBro-Birds/MechaCar_Statistical_Analysis/master/support_images_read.me/t_test_lot3.png)

* The t-test results for all vehicles and lots 1 & 2 show that these are not statistically different since each has a p-value > 0.05.  Lot 3 however is statistically different since the p-value of 0.04 > 0.05.

## Study Design: MechaCar vs Competition
Today's political environment and climate change factors are creating high volatility in fuel cost.  This in turn has a large percentage of automobile consumers focused on mpg when purchasing new vehicles.  Consumer purchusing preference also varies based on needed vehicle size.  The metrics to test to see how MechaCar performs against the competition is city mpg and highway mpg by vehicle size class.  The data needed by vehicle type to perform the analysis would be city mpg, highway mpg, vehicle dimensions (length, width, height, clearance), vehicle weight and vehicle cargo load.  Two multiple linear regressions should be performed; one for city mpg and one for highway mpg.  The dependent variable would be the mpg and the other data attributes would be the independent variables.  The multiple linear regression results would assist management and development teams in producing vehicles meeting the desired specifications of the consumer.      
