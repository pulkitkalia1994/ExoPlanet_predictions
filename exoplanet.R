###Reading the data
data<-read.csv("C:\\R-Programming\\Exoplanet\\oec.csv")

library("reshape2")
library("dplyr")
library(ggplot2)
data$PlanetIdentifier<-NULL
##Plotting some graphs for explorations
df.m = melt(data, id.var="TypeFlag")
g<-ggplot(df.m,aes(TypeFlag,value))+geom_point()+facet_wrap(~variable,ncol=4)
print(g)

temp<-data
temp$PlanetIdentifier<-NULL
library("caret")
library("caTools")
###Splitting the data into training and testing set
split<-sample.split(temp$TypeFlag,SplitRatio = 0.8)
train<-subset(temp,split==TRUE)
test<-subset(temp,split==FALSE)
temptest<-test
test$TypeFlag<-NULL

library("xgboost")
library("Matrix")
library(e1071)
library("dplyr")
##Convering factor variables to numeric type
train<- train %>%  mutate_if(is.factor,as.numeric)
test<- test %>%  mutate_if(is.factor,as.numeric)

##Excluding the target variable(dependent variable)
data_variables <- as.matrix(train[,-1])
data_label <- train[,"TypeFlag"]
data_matrix <- xgb.DMatrix(data = data_variables, label = data_label)

numberOfClasses <- length(unique(train$TypeFlag))
xgb_params <- list("objective" = "multi:softmax",eta=0.04,gamma=0.6,max_depth=10,eval_metric="mlogloss","num_class" = numberOfClasses)
xgbcv <- xgb.cv( params = xgb_params, data = data_matrix, nrounds = 400, nfold = 10, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)


nround    <- xgbcv$best_iteration # number of XGBoost rounds
cv.nfold  <- 10

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
bst_model <- xgb.train(params = xgb_params,
                       data = data_matrix,
                       nrounds = nround)

test_matrix<-xgb.DMatrix(data = as.matrix(test))
predictionsXGBoost<-predict(bst_model,newdata=test_matrix)

table(predictionsXGBoost,temptest$TypeFlag)
##Getting an accuracy of 99.86% which is huge! (Alternatively, you can use logistic regression and use a less threshhold to increase the sensitivity but by doing this the accuracy will decrease.)

