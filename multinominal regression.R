############################################################
# Title:        Multinominomal Regression
# Author:       Sumukh Umesha
# Date:         03/18/2018
# Description:  Training & validation of the dataset
############################################################

rm(list=ls(all=TRUE))

## Import packages
library(data.table)
library(sandwich) 
library(lmtest)
library(tseries)
library(plyr)
library(expm)
library(MASS)
library(boot)
library(plm)
library(rJava)
library(xlsx)
library(cvTools)
library(caret)
library(caTools)
library(foreign)
library(Hmisc)
library(mlogit)
library(nnet)



#------------

#Reading the entire dataset
context1 <- fread('TrainingDataset.csv')

#Reading data with weekly earnings above 0
context4 <- fread('trainingabove0.csv')


# COnverting numerics to factors
context4$EducationLevel <- factor(context4$EducationLevel, levels=sort(unique(context4$EducationLevel)))
context4$Age <- factor(context4$Age, levels=sort(unique(context4$Age)))
context4$EmploymentStatus <- factor(context4$EmploymentStatus, levels=sort(unique(context4$EmploymentStatus)))
context4$Gender <- factor(context4$Gender, levels=sort(unique(context4$Gender)))
context4$Children <- factor(context4$Children, levels=sort(unique(context4$Children)))
context4$WeeklyEarnings <- factor(context4$WeeklyEarnings, levels=sort(unique(context4$WeeklyEarnings)))
context4$WeeklyHours <- factor(context4$WeeklyHours, levels=sort(unique(context4$WeeklyHours)))
context4$Sleeping <- factor(context4$Sleeping, levels=sort(unique(context4$Sleeping)))
context4$Grooming <- factor(context4$Grooming, levels=sort(unique(context4$Grooming)))
context4$Housework <- factor(context4$Housework, levels=sort(unique(context4$Housework)))
context4$FoodDrink <- factor(context4$FoodDrink, levels=sort(unique(context4$FoodDrink)))
context4$CarForChi <- factor(context4$CarForChi, levels=sort(unique(context4$CarForChi)))
context4$PlaywChi <- factor(context4$PlaywChi, levels=sort(unique(context4$PlaywChi)))
context4$JobSearching <- factor(context4$JobSearching, levels=sort(unique(context4$JobSearching)))
context4$Shopping <- factor(context4$Shopping, levels=sort(unique(context4$Shopping)))
context4$EatDrink <- factor(context4$EatDrink, levels=sort(unique(context4$EatDrink)))
context4$Socializing <- factor(context4$Socializing, levels=sort(unique(context4$Socializing)))
context4$Television <- factor(context4$Television, levels=sort(unique(context4$Television)))
context4$Golfing <- factor(context4$Golfing, levels=sort(unique(context4$Golfing)))
context4$Running <- factor(context4$Running, levels=sort(unique(context4$Running)))


#spliting the test dataset into 70% training & 30% validation
context4$spl=sample.split(context4,SplitRatio=0.7)
View(context4)

#training data set
trainabv0=subset(context4, context4$spl==TRUE)
View(trainabv0)

#test data set
testabv0=subset(context4, context4$spl==FALSE)
View(testabv0)


# reading data with weekly earings equal to 0 
context3 <- fread('training=0.csv')

#numerics to factors
context3$EducationLevel <- factor(context3$EducationLevel, levels=sort(unique(context3$EducationLevel)))
context3$Age <- factor(context3$Age, levels=sort(unique(context3$Age)))
context3$EmploymentStatus <- factor(context3$EmploymentStatus, levels=sort(unique(context3$EmploymentStatus)))
context3$Gender <- factor(context3$Gender, levels=sort(unique(context3$Gender)))
context3$Children <- factor(context3$Children, levels=sort(unique(context3$Children)))
context3$WeeklyEarnings <- factor(context3$WeeklyEarnings, levels=sort(unique(context3$WeeklyEarnings)))
context3$WeeklyHours <- factor(context3$WeeklyHours, levels=sort(unique(context3$WeeklyHours)))
context3$Sleeping <- factor(context3$Sleeping, levels=sort(unique(context3$Sleeping)))
context3$Grooming <- factor(context3$Grooming, levels=sort(unique(context3$Grooming)))
context3$Housework <- factor(context3$Housework, levels=sort(unique(context3$Housework)))
context3$FoodDrink <- factor(context3$FoodDrink, levels=sort(unique(context3$FoodDrink)))
context3$CarForChi <- factor(context3$CarForChi, levels=sort(unique(context3$CarForChi)))
context3$PlaywChi <- factor(context3$PlaywChi, levels=sort(unique(context3$PlaywChi)))
context3$JobSearching <- factor(context3$JobSearching, levels=sort(unique(context3$JobSearching)))
context3$Shopping <- factor(context3$Shopping, levels=sort(unique(context3$Shopping)))
context3$EatDrink <- factor(context3$EatDrink, levels=sort(unique(context3$EatDrink)))
context3$Socializing <- factor(context3$Socializing, levels=sort(unique(context3$Socializing)))
context3$Television <- factor(context3$Television, levels=sort(unique(context3$Television)))
context3$Golfing <- factor(context3$Golfing, levels=sort(unique(context3$Golfing)))
context3$Running <- factor(context3$Running, levels=sort(unique(context3$Running)))
context3$Volunteering <- factor(context3$Volunteering, levels=sort(unique(context3$Volunteering)))


#splitting the dataset into training & testing
context3$spl=sample.split(context3,SplitRatio=0.7)
View(context3)

#training data set
train1=subset(context3, context3$spl==TRUE)
View(train1)

#test data set
test1=subset(context3, context3$spl==FALSE)
View(test1)

#Multinominal regression model
multinommodel1 <- multinom(EmploymentStatus ~ EducationLevel+	Age+	Gender	+	WeeklyHours	+	JobSearching	+Socializing, data = train1, family="multinomial", MaxNWts =10000000)
#summary(multinommodel1)  (takes times to runs as the numbers of weights is heavy. The significant variables are tested on logistic regression model)

#predicted probability
predicted_scores <- predict(multinommodel1, test1, "probs")
View(predicted_scores)

#predicted classes of output variable
predicted_class <- predict(multinommodel1, test1)
View(predicted_class)


#Data type alterations
y1 <- (as.integer(unlist(data.frame(test1$EmploymentStatus))))
View(y1)


#Confusion matrix
confusionMatrix(y1,predicted_class)
# Confusion Matrix and Statistics
# 
#           Reference
# Prediction    1    2    3
# 1           196  895    6
# 2           158 6409   18
# 3            11  187 1348
# 
# Overall Statistics
# 
# Accuracy : 0.8618          
# 95% CI : (0.8546, 0.8688)
# No Information Rate : 0.8118          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.6467          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity           0.53699   0.8556   0.9825
# Specificity           0.89834   0.8987   0.9748
# Pos Pred Value        0.17867   0.9733   0.8719
# Neg Pred Value        0.97922   0.5906   0.9969
# Prevalence            0.03955   0.8118   0.1487
# Detection Rate        0.02124   0.6945   0.1461
# Detection Prevalence  0.11888   0.7136   0.1675
# Balanced Accuracy     0.71766   0.8771   0.9787

#misclassfication error
mean(as.character(predicted_class) != as.character(test1$EmploymentStatus)) #0.1381664, 13.81% misclassifcation error.


#testing the model for whole data set
predictedval <- data.frame(predicted_class)
View(predictedval)

y2 <- (as.integer(unlist(data.frame(testabv0$EmploymentStatus))))
y2datframe <- data.frame(y2)
colnames(y2datframe) <- "predicted_class"

View(y2datframe)
finalpredicted <- rbind(predictedval, y2datframe)
predictedvector <- (as.integer(unlist(data.frame(finalpredicted$predicted_class))))

y1 <- append(y1,y2)

confusionMatrix(predictedvector,y1)
# Confusion Matrix and Statistics
# 
#           Reference
# Prediction     1     2     3
#          1 13184   158    11
#          2   895  6409   187
#          3     6    18  1348
# 
# Overall Statistics
# 
# Accuracy : 0.9426          
# 95% CI : (0.9395, 0.9456)
# No Information Rate : 0.634           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8885          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Statistics by Class:
# 
# Class: 1 Class: 2 Class: 3
# Sensitivity            0.9360   0.9733  0.87193
# Specificity            0.9792   0.9308  0.99884
# Pos Pred Value         0.9873   0.8556  0.98251
# Neg Pred Value         0.8983   0.9880  0.99050
# Prevalence             0.6340   0.2964  0.06959
# Detection Rate         0.5934   0.2885  0.06068
# Detection Prevalence   0.6011   0.3372  0.06176
# Balanced Accuracy      0.9576   0.9520  0.93538
# recall for class 1- 98.7
# recall for class 2- 85.86
# recall for class 3 - 98.2
# Average recall value is 94.2

mean(as.character(predictedvector) != as.character(y1)) #0.05739107. 5.7% error

# Predictions on the test dataset
testdataset <- fread('TestDataset.csv')

#converting numerics to factors
testdataset$EducationLevel <- factor(testdataset$EducationLevel, levels=sort(unique(testdataset$EducationLevel)))
testdataset$Age <- factor(testdataset$Age, levels=sort(unique(testdataset$Age)))
testdataset$EmploymentStatus <- factor(testdataset$EmploymentStatus, levels=sort(unique(testdataset$EmploymentStatus)))
testdataset$Gender <- factor(testdataset$Gender, levels=sort(unique(testdataset$Gender)))
testdataset$Children <- factor(testdataset$Children, levels=sort(unique(testdataset$Children)))
testdataset$WeeklyEarnings <- factor(testdataset$WeeklyEarnings, levels=sort(unique(testdataset$WeeklyEarnings)))
testdataset$WeeklyHours <- factor(testdataset$WeeklyHours, levels=sort(unique(testdataset$WeeklyHours)))
testdataset$Sleeping <- factor(testdataset$Sleeping, levels=sort(unique(testdataset$Sleeping)))
testdataset$Grooming <- factor(testdataset$Grooming, levels=sort(unique(testdataset$Grooming)))
testdataset$Housework <- factor(testdataset$Housework, levels=sort(unique(testdataset$Housework)))
testdataset$FoodDrink <- factor(testdataset$FoodDrink, levels=sort(unique(testdataset$FoodDrink)))
testdataset$CarForChi <- factor(testdataset$CarForChi, levels=sort(unique(testdataset$CarForChi)))
testdataset$PlaywChi <- factor(testdataset$PlaywChi, levels=sort(unique(testdataset$PlaywChi)))
testdataset$JobSearching <- factor(testdataset$JobSearching, levels=sort(unique(testdataset$JobSearching)))
testdataset$Shopping <- factor(testdataset$Shopping, levels=sort(unique(testdataset$Shopping)))
testdataset$EatDrink <- factor(testdataset$EatDrink, levels=sort(unique(testdataset$EatDrink)))
testdataset$Socializing <- factor(testdataset$Socializing, levels=sort(unique(testdataset$Socializing)))
testdataset$Television <- factor(testdataset$Television, levels=sort(unique(testdataset$Television)))
testdataset$Golfing <- factor(testdataset$Golfing, levels=sort(unique(testdataset$Golfing)))
testdataset$Running <- factor(testdataset$Running, levels=sort(unique(testdataset$Running)))


#predict
yhat <- predict(multinommodel1, testdataset, se.fit =FALSE)


# EducationLevel	Age	EmploymentStatus	Gender	Children	WeeklyEarnings	WeeklyHours	Sleeping	Grooming	Housework	FoodDrink	CarForChi	PlaywChi	JobSearching	Shopping	EatDrink	Socializing	Television	Golfing	Running	Volunteering	Total
