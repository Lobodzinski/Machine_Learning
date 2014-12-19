### 1. Introduction:

description of the project:
Project is a part of the exercises for the course: Practical Machine Learning from Johns Hopkins University
provided by: Jeff Leek, PhD, Roger D. Peng, PhD, Brian Caffo, PhD
Detailed description of the task one can find on the URL:
https://class.coursera.org/predmachlearn-016/human_grading/view/courses/973763/assessments/4/submissions

Detailed description of the project the reader can find in the work [1]

#### Access to the data:

The URL to the training data:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The URL to the test data:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data can be also find in the source of the project: http://groupware.les.inf.puc-rio.br/har.


### Tools:
according to the article: 
"Qualitative Activity Recognition of Weight Lifting Exercises" (Eduardo Velloso,
Andreas Bulling, Hans Gellersen, Wallace Ugulino & Hugo Fuks)
available on the URL: http://groupware.les.inf.puc-rio.br/har 
we use: 

A) Random Forest approach 
- for description with references therin: http://en.wikipedia.org/wiki/Random_forest,
- examples of application in r language: 
r-package: 	http://cran.r-project.org/web/packages/randomForest/index.html,
examples: 	http://www.statmethods.net/advstats/cart.html

B) Bootstrap aggregating (bagging) method
r-package:	http://www.inside-r.org/packages/cran/ipred/docs/bagging
examples:	http://www.vikparuchuri.com/blog/intro-to-ensemble-learning-in-r/	<---

C) Tree-based prediction method (CART)
r-package: rpart


### Analysis:
#### 1. how you built your model,

##### a) initial seps:
set.seed(123456)
library(caret)
library(RCurl)

Download of the training and testing data:

train<-read.csv(textConnection(getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")))
test<-read.csv(textConnection(getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")))

##### b) Clean-up the data:

###### step 1: 

using 
featurePlot(x = train[, names1], y = train$classe, plot = "pairs") 
where names1 is a part of list of general list 
names(train) I selected a few columns which can be removed from the train part. 
Due to the limited RAM, I moved through all column names selecting for each check 10 column names.
I noticed negative response in case of columns:

c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))   

The removal of unrelevant column is:

train1<-subset(train, select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))

###### step 2:

check NearZeroVariance variables: 

nearZeroVar(train1, saveMetrics=TRUE)

I selected all columns with nzv=TRUE and removed from the train set:

train2<-subset(train1, select=-c(kurtosis_roll_belt,kurtosis_picth_belt,kurtosis_yaw_belt,skewness_roll_belt,skewness_roll_belt.1,skewness_yaw_belt,max_yaw_belt,
min_yaw_belt,amplitude_yaw_belt,kurtosis_roll_arm,kurtosis_picth_arm,kurtosis_yaw_arm,skewness_roll_arm,skewness_pitch_arm,skewness_yaw_arm,
kurtosis_roll_dumbbell,kurtosis_picth_dumbbell,kurtosis_yaw_dumbbell,skewness_roll_dumbbell,skewness_pitch_dumbbell,skewness_yaw_dumbbell,
max_yaw_dumbbell,min_yaw_dumbbell,amplitude_yaw_dumbbell,kurtosis_roll_forearm,kurtosis_picth_forearm,kurtosis_yaw_forearm,skewness_roll_forearm,
skewness_pitch_forearm,skewness_yaw_forearm,max_yaw_forearm,min_yaw_forearm,amplitude_yaw_forearm))

dim(train2)

19622    53

##### c) splitting the data:

training set: trainSet=60 %; 
test set: testSet=40 %;

inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
trainSet <- train1[inTrain, ]
testSet <- train1[-inTrain, ]
dim(trainSet)
dim(testSet)


#### 2. how you used cross validation:

The first prediction method (Tree based) is used as a main algorithm.
The cross-validation is performed using the Random Forest and the bagging methods. 
Having all outputs I compare predictions.

##### a) Tree-based prediction mode:

library(rpart)
m1 <- rpart(classe ~ ., data = trainSet)
library(rattle)
fancyRpartPlot(m1)

The plot: 
![Settings Window](https://github.com/Lobodzinski/Machine_Lerning/TreeRplot_Fig1.png)

#### 3. what you think the expected out of sample error is, and

#### 4. why you made the choices you did.
In terms of the selection of prediction methods, I decided to use the Random Tree and the Bagging algorithms 
because both methods are used for analysis of the data in the source article [1]. 
The Tree based prediction is used due to the lectures.  

### Summary:


### References:
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
"Qualitative Activity Recognition of Weight Lifting Exercises."
Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


