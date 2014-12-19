### 1. Introduction:

Author: 		Bogdan Lobodzinski
GitHub repo:		https://github.com/Lobodzinski/Machine_Learning

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
examples:	http://www.vikparuchuri.com/blog/intro-to-ensemble-learning-in-r/

C) Tree-based prediction method (CART)
r-package: rpart


### Analysis:
#### 1. how you built your model,

##### a) initial seps:
```
set.seed(123456)
library(caret)
library(RCurl)
```

Download of the training and testing data:

```
train<-read.csv(textConnection(getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")))
test<-read.csv(textConnection(getURL("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")))
```

##### b) Clean-up the data:

###### step 1: 

using 

`featurePlot(x = train[, names1], y = train$classe, plot = "pairs") `
where names1 is a part of list of general list 

`names(train)` I selected a few columns which can be removed from the train part. 
Due to the limited RAM, I moved through all column names selecting for each check 10 column names.
I noticed negative response in case of columns:

`c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))   `

The removal of unrelevant column is:

`train1<-subset(train, select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))`

###### step 2:

check NearZeroVariance variables: 

`nearZeroVar(train1, saveMetrics=TRUE)`

I selected all columns with `nzv=TRUE` and removed from the train set:

```
train2<-subset(train1, select=-c(kurtosis_roll_belt,kurtosis_picth_belt,kurtosis_yaw_belt,skewness_roll_belt,skewness_roll_belt.1,skewness_yaw_belt,max_yaw_belt,
min_yaw_belt,amplitude_yaw_belt,kurtosis_roll_arm,kurtosis_picth_arm,kurtosis_yaw_arm,skewness_roll_arm,skewness_pitch_arm,skewness_yaw_arm,
kurtosis_roll_dumbbell,kurtosis_picth_dumbbell,kurtosis_yaw_dumbbell,skewness_roll_dumbbell,skewness_pitch_dumbbell,skewness_yaw_dumbbell,
max_yaw_dumbbell,min_yaw_dumbbell,amplitude_yaw_dumbbell,kurtosis_roll_forearm,kurtosis_picth_forearm,kurtosis_yaw_forearm,skewness_roll_forearm,
skewness_pitch_forearm,skewness_yaw_forearm,max_yaw_forearm,min_yaw_forearm,amplitude_yaw_forearm))`
dim(train2)
[1] 19622    53
```

##### c) splitting the data:

training set: trainSet=60 %; 
test set: testSet=40 %;

```
inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
trainSet <- train1[inTrain, ]
testSet <- train1[-inTrain, ]
dim(trainSet)
dim(testSet)
```


#### 2. how you used cross validation:

The first prediction method (Tree based) is used as a main algorithm.
The cross-validation is performed using the Random Forest and the bagging methods. 
Having all outputs I compare predictions.

##### a) The Tree-based prediction model:

```
library(rpart)
m1 <- rpart(classe ~ ., data = trainSet)
library(rattle)
fancyRpartPlot(m1)
```

The plot: 
![](https://github.com/Lobodzinski/Machine_Learning/blob/master/TreeRplot_Fig1.png)

##### b) The Random Forest prediction model:
```
library(randomForest)
m2 <- randomForest(classe ~ ., data = trainSet, importance = T)
m2
plot(m2)
```
![](https://github.com/Lobodzinski/Machine_Learning/blob/master/ForestRplot_Fig2.png)
```
barplot(m2$importance[, 7], main = "Gini coeff")
```
![](https://github.com/Lobodzinski/Machine_Learning/blob/master/GiniRplot_Fig3.png)

##### c) The Bagging prediction model:
```
library(ipred)
m3 <- bagging(classe ~ ., data = train2, coob = T)
print(m3)

Bagging classification trees with 25 bootstrap replications 

Call: bagging.data.frame(formula = classe ~ ., data = train2, coob = T)

Out-of-bag estimate of misclassification error:  0.0145 

```
##### d) Comparison of the prediction methods: 
```
Tree <- predict(m1, testSet, type = "class")
Forest <- predict(m2, testSet)
Bagging <- predict(m3, testSet)
```

For comparison one can use the `confusionMatrix` and 
the function `errorest` (from `ipred` package). In the last case be patient, 
the `errorest` is a very long time consuming process. I show both error estimations.

###### confusionMatrix:

for the Tree-based method: 
```
treeconf<-confusionMatrix(Tree, testSet$classe)
treeconf

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2024  256   82  151   52
         B   71  923  148  116  144
         C   69  169 1039  191  161
         D   45  141   99  722   79
         E   23   29    0  106 1006

Overall Statistics

               Accuracy : 0.7283
                 95% CI : (0.7183, 0.7381)
    No Information Rate : 0.2845
    P-Value [Acc > NIR] : < 2.2e-16

                  Kappa : 0.6544
 Mcnemar's Test P-Value : < 2.2e-16

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9068   0.6080   0.7595  0.56143   0.6976
Specificity            0.9036   0.9243   0.9089  0.94451   0.9753
Pos Pred Value         0.7891   0.6583   0.6378  0.66483   0.8643
Neg Pred Value         0.9606   0.9077   0.9471  0.91657   0.9348
Prevalence             0.2845   0.1935   0.1744  0.16391   0.1838
Detection Rate         0.2580   0.1176   0.1324  0.09202   0.1282
Detection Prevalence   0.3269   0.1787   0.2076  0.13841   0.1484
Balanced Accuracy      0.9052   0.7662   0.8342  0.75297   0.8365

``` 

for the Random Forest method:
```
forestconf<-confusionMatrix(Forest, testSet$classe)
forestconf

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2232    9    0    0    0
         B    0 1505    8    0    0
         C    0    4 1359   23    2
         D    0    0    1 1263    2
         E    0    0    0    0 1438

Overall Statistics

               Accuracy : 0.9938
                 95% CI : (0.9918, 0.9954)
    No Information Rate : 0.2845
    P-Value [Acc > NIR] : < 2.2e-16

                  Kappa : 0.9921
 Mcnemar's Test P-Value : NA

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   0.9914   0.9934   0.9821   0.9972
Specificity            0.9984   0.9987   0.9955   0.9995   1.0000
Pos Pred Value         0.9960   0.9947   0.9791   0.9976   1.0000
Neg Pred Value         1.0000   0.9979   0.9986   0.9965   0.9994
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2845   0.1918   0.1732   0.1610   0.1833
Detection Prevalence   0.2856   0.1928   0.1769   0.1614   0.1833
Balanced Accuracy      0.9992   0.9951   0.9945   0.9908   0.9986

```

for the Bagging method:
```
baggingconf<-confusionMatrix(Bagging, testSet$classe)
baggingconf

Confusion Matrix and Statistics
          Reference
Prediction    A    B    C    D    E
         A 2232    0    0    0    0
         B    0 1518    0    0    0
         C    0    0 1368    3    2
         D    0    0    0 1283    0
         E    0    0    0    0 1440

Overall Statistics

               Accuracy : 0.9994
                 95% CI : (0.9985, 0.9998)
    No Information Rate : 0.2845
    P-Value [Acc > NIR] : < 2.2e-16

                  Kappa : 0.9992
 Mcnemar's Test P-Value : NA

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   0.9977   0.9986
Specificity            1.0000   1.0000   0.9992   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   0.9964   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   0.9995   0.9997
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2845   0.1935   0.1744   0.1635   0.1835
Detection Prevalence   0.2845   0.1935   0.1750   0.1635   0.1835
Balanced Accuracy      1.0000   1.0000   0.9996   0.9988   0.9993

```
So, we have:
```
		Accuracy:	

Tree:		0.7283
Forest:		0.9938
Bagging:	0.9994
```

According to confusionMatrix the best prediction method is the `Bagging`. 

###### errorest:

The check is started as:
```
mypredict.rpart <- function(object, newdata) 
{
	predict(object, newdata = newdata, type = "class")
}

errorestRes<-c(Tree = errorest(classe ~ ., data = testSet, model = rpart, predict = mypredict.rpart)$error,
Bagging = errorest(classe ~ ., data = testSet, model = bagging)$error,
Forest = errorest(classe ~ ., data = testSet, model = randomForest)$error)

      Tree    Bagging     Forest
0.24751466 0.02702014 0.01261789
```

According to the errorest the better prediction algorithm is the Forest Tree. 

#### 3. what you think the expected out of sample error is / Conclusions:
Calculation of the confusion Matrix gives accuracy which points to the Random Forest prediction method as the best
choice for the above task.
However, according to the function errorest (from ipred package), the best prediction method is the Bagging.
Using my knowledge I cannot distinguish which method is a proper one:
the Random Forest or the Bagging. A more sophisticated check is necessary.
So, I can only write, that the Tree prediction method shows the largest error and should not be used in this 
kind of analysis.

It is unclear to me how to use the testing data (from
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv ).
It cannot be added to the testSet because the column "classe" is missing, therefore we cannot compare our predictions with 
the testing data .

#### 4. why you made the choices you did.
In terms of the selection of prediction methods, I decided to use the Random Tree and the Bagging algorithms 
because both methods are used for analysis of the data in the source article [1]. 
The Tree based prediction is used due to the lectures.  

### References:
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
"Qualitative Activity Recognition of Weight Lifting Exercises."
Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


