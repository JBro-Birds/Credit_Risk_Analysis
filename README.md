# Credit_Risk_Analysis

## Overview of Project
In 2019 there was a record-breaking number of more than 19 million Americans with at least one unsecured personal loan. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.  To dive into understanding machine learning techniques Jill, my data scientist boss, has asked me to research the difference between regression and classification models to predict credit risk.  Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.    

For this analysis I'll use Python and Scikit-learn, a Python machine learning library, to build and evaluate several machine learning models to predict credit risk.  The first model I'll use is logistic regression, a popular classification model.  With logistic regression, it is possible to try to answer questions such as whether a credit card holder is likely to miss a payment in the next month.  I'll follow the familiar pattern as I instantiate a model, train it, create predictions, then validate the model.  The model must answer important questions such as how well does it perform? what percentage of predictions does it get right?  These questions will be answered by performing accuracy scoring and metrics covering sensitivity and precision.  Next, to improve the model's performance Jill has asked me to perform ensemble learning, random forest modeling and resampling techniques.

### Purpose
Jill has requested I perform a deep dive into the resampling aspect of machine learning to further enhance credit risk modeling.  Due to credit risk being an inherently unbalanced classification problem, as good loans easily outnumber risky loans, Jill has aksed me to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.  Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once completed, I’ll evaluate the performance of these models and write a recommendation on whether they should be used to predict credit risk.

## Results

*  Naive Random Oversampling:  accuracy score is 65.7%; precision score is a dismal 1% for high-risk but an outstanding rate of 100% for low-risk; recall score is 71% for high-risk and 60% for low-risk.

![NR_BalancedAccuracyScore](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/NR_BalancedAccuracyScore.png)
![NR_ImbalancedClassReport](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/NR_ImbalancedClassReport.png)

*  SMOTE Oversampling:  accuracy score is 66.2%; precision score is a dismal 1% for high-risk but an outstanding rate of 100% for low-risk; recall score is 63% for high-risk and 69% for low-risk.

![Smote_BalancedAccuracyScore](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/Smote_BalancedAccuracyScore.png)
![Smote_ImbalancedClassReport](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/Smote_ImbalancedClassReport.png)

*  Undersampling:  accuracy score on lower side at 54.4%; precision score is a dismal 1% for high-risk but an outstanding rate of 100% for low-risk; recall score is 69% for high-risk and 40% for low-risk.

![Undersampling_BalancedAccuracyScore](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/Undersampling_BalancedAccuracyScore.png)
![Undersampling_ImbalancedClassReport](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/Undersampling_ImbalancedClassReport.png)

*  Combination (Over and Under) Sampling:  accuracy score is 68.8%; precision score is a dismal 1% for high-risk but an outstanding rate of 100% for low-risk; recall score is 80% for high-risk and 57% for low-risk.

![Combo_BalancedAccuracyScore](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/Combo_BalancedAccuracyScore.png)
![Combo_ImbalancedClassReport](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/Combo_ImbalancedClassReport.png)

*  Balanced Random Forest Classifier:  accuracy score is 78.9%; precision score is a dismal 3% for high-risk but an outstanding rate of 100% for low-risk; recall score is 70% for high-risk and 87% for low-risk.

![RandomForest_BalancedAccuracyScore](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/RandomForest_BalancedAccuracyScore.png)
![RandomForest_ImbalancedClassReport](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/RandomForest_ImbalancedClassReport.png)

*  Easy Ensemble AdaBoost Classifier:  accuracy score is on higher side at 93.2%; precision score is a dismal 9% for high-risk but an outstanding rate of 100% for low-risk; recall score is 92% for high-risk and 94% for low-risk.

![EasyEnsemble_BalancedAccuracyScore](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/EasyEnsemble_BalancedAccuracyScore.png)
![EasyEnsemble_ImbalancedClassReport](https://raw.githubusercontent.com/JBro-Birds/Credit_Risk_Analysis/master/support_readme_images/EasyEnsemble_ImbalancedClassReport.png)

## Summary
Comparing the results of all six models shows that there are some major differences in prediction accuracy.  I recommend using the Easy Ensemble AdaBoost Classifier for a couple of reasons.  First, the balanced accuracy is 93% which is much higher that the other models.  Second, the recall score is 92% for loan applicants identifed as high-risk.  This too is much higher compared to the other models for the category of high-risk.  The recall (sensitivity) metric is important in predicting credit risk because lending companies would want to flag (predict) as many high-risk loan applicants as possible.  For the high-risk category ("actually true") the Easy Ensemble AdaBoost Classifier model accurately predicted 93% as high-risk ("true positive") vs. only 7% predicted as not high-risk (false negative).  For financial reasons for lending companies they would want a model that has a sensitivity score for high-risk applicants to be as close to 100% as possible.    

 
