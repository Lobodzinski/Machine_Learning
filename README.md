
### Introduction:

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

For analysis of the data [1] I use:

A) Tree-based prediction method (CART)
r-package: rpart

B) Random Forest approach 
- for description with references therin: http://en.wikipedia.org/wiki/Random_forest,
- examples of application in r language: 
r-package: 	http://cran.r-project.org/web/packages/randomForest/index.html,
examples: 	http://www.statmethods.net/advstats/cart.html

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

```
train1<-subset(train, select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))
test1<-subset(test, select=-c(X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window))
````

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

test2<-subset(test1, select=-c(kurtosis_roll_belt,kurtosis_picth_belt,kurtosis_yaw_belt,skewness_roll_belt,skewness_roll_belt.1,skewness_yaw_belt,max_yaw_belt,
min_yaw_belt,amplitude_yaw_belt,kurtosis_roll_arm,kurtosis_picth_arm,kurtosis_yaw_arm,skewness_roll_arm,skewness_pitch_arm,skewness_yaw_arm,
kurtosis_roll_dumbbell,kurtosis_picth_dumbbell,kurtosis_yaw_dumbbell,skewness_roll_dumbbell,skewness_pitch_dumbbell,skewness_yaw_dumbbell,
max_yaw_dumbbell,min_yaw_dumbbell,amplitude_yaw_dumbbell,kurtosis_roll_forearm,kurtosis_picth_forearm,kurtosis_yaw_forearm,skewness_roll_forearm,
skewness_pitch_forearm,skewness_yaw_forearm,max_yaw_forearm,min_yaw_forearm,amplitude_yaw_forearm))`

dim(test2)
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


#### 2. Predictions / how you used cross validation:

##### a) The Tree-based prediction model:

```
library(rpart)
m1 <- rpart(classe ~ ., data = trainSet)
library(rattle)
fancyRpartPlot(m1)
```
![](https://github.com/Lobodzinski/Machine_Learning/blob/master/TreeRplot_Fig1.png)

##### b) The Random Forest prediction model:
The 5-fold cross-validation is performed for the Random Forest prediction method
(http://topepo.github.io/caret/training.html).

```
fitControl <- trainControl(## 5-fold CV
                           method = "repeatedcv",
                           number = 5,
                           ## repeated ten times
                           repeats = 5)

m2 <- train(classe ~ ., data = trainSet,
                 method = "rf",
                 trControl = fitControl,
                 verbose = FALSE)
```

##### c) Comparison of the prediction methods:

I compare both prediction method using `confusionMatrix`.

For the Tree-based prediction algorithm:
```
Tree <- predict(m1, testSet, type = "class")
table(testSet$classe, Tree)

   Tree
       A    B    C    D    E
  A 2031   79   65   25   32
  B  212  960  139  138   69
  C   21  107 1112   88   40
  D   68   94  214  830   80
  E   19  140  168   81 1034

Treeconf<-confusionMatrix(Tree, testSet$classe)
Treeconf

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2031  212   21   68   19
         B   79  960  107   94  140
         C   65  139 1112  214  168
         D   25  138   88  830   81
         E   32   69   40   80 1034

Overall Statistics
                                          
               Accuracy : 0.7605          
                 95% CI : (0.7509, 0.7699)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6966          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9099   0.6324   0.8129   0.6454   0.7171
Specificity            0.9430   0.9336   0.9095   0.9494   0.9655
Pos Pred Value         0.8639   0.6957   0.6549   0.7143   0.8239
Neg Pred Value         0.9634   0.9137   0.9584   0.9318   0.9381
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2589   0.1224   0.1417   0.1058   0.1318
Detection Prevalence   0.2996   0.1759   0.2164   0.1481   0.1600
Balanced Accuracy      0.9265   0.7830   0.8612   0.7974   0.8413
```

In case of the Random Forest method:
```
Forest <- predict(m2, testSet)
forestconf<-confusionMatrix(Forest, testSet$classe)
forestconf

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2231   16    0    0    1
         B    1 1497    8    0    2
         C    0    5 1351   22    4
         D    0    0    9 1264    3
         E    0    0    0    0 1432

Overall Statistics
                                          
               Accuracy : 0.991           
                 95% CI : (0.9886, 0.9929)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9886          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9996   0.9862   0.9876   0.9829   0.9931
Specificity            0.9970   0.9983   0.9952   0.9982   1.0000
Pos Pred Value         0.9924   0.9927   0.9776   0.9906   1.0000
Neg Pred Value         0.9998   0.9967   0.9974   0.9967   0.9984
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2843   0.1908   0.1722   0.1611   0.1825
Detection Prevalence   0.2865   0.1922   0.1761   0.1626   0.1825
Balanced Accuracy      0.9983   0.9922   0.9914   0.9905   0.9965
```

So, I got:
```
		Accuracy:	

Tree	:	0.7605
Forest	:	0.991
```

The Random Forest method is more reliable then the Tree-based algorithm. Therefore for the 
submission part of the project I will use the The Random Forest method.
Results based on the test data:
```
forestFit <- predict(m2$finalModel, newdata = test2)
forestFit
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
Levels: A B C D E
```

##### Wrinting result to the files:

on the Linux OS I used the following way. 
```
pml_write_files = function(x){
     n = length(x)
     for(i in 1:n){
         filename = paste0("submission/problem_id_",i,".txt")
         write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
     }
}
 
pml_write_files(forestFit)
```

After creation of files, removal of a new line sign inside each file is necessary.
It can be done using `zsh` loop:

```
foreach i (`ls`)
echo $i
perl -i -pe 's/\n//g' $i
end
```
The size of each file `problem_id_*.txt` should be 1 . 

#### 3. what you think the expected out of sample error is / Conclusions:
The Random Forests prediction method generates better results then the Tree-based algorithm.

#### 4. why you made the choices you did.
The Tree-based and the Random Forest prediction algorithms are more or less described in available lectures. 
In addition, the Random Forest method is used for the alalysis in the source article [1]. 

### References:
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
"Qualitative Activity Recognition of Weight Lifting Exercises."
Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
(available on http://groupware.les.inf.puc-rio.br/har)


