# libraries
library(dplyr)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(VIM)
library(gridExtra)
library(magrittr)
library(naniar)

# Loading training data
traindata=read.csv('Train.csv')
head(traindata)
View(traindata)

dataset<-traindata
View(dataset)
-------------------------------# Data preprocessing ------------------------------------

# Transforming "low fat" and "LF" to "Low Fat" and transforming "reg" to "Regular"
dataset$Item_Fat_Content=
  factor(dataset$Item_Fat_Content,
         levels =c('LF','Low Fat','low fat','Reg','Regular' ),
         labels = c('Low Fat','Low Fat','Low Fat','Regular','Regular'))

# missing values imputation for Item_Weight (Replacing na values by mean)
dataset$Item_Weight=
  ifelse(is.na(dataset$Item_Weight),
         ave(dataset$Item_Weight,FUN=function(x)
           mean(x,na.rm = TRUE)),dataset$Item_Weight)

# Imputing the missing values for Outlet_Size (Replacing by Mode of Outlet_Size which is "Medium")
table(dataset$Outlet_Identifier, dataset$Outlet_Size)
table(dataset$Outlet_Identifier, dataset$Outlet_Type)
table(dataset$Outlet_Type, dataset$Outlet_Size)

index1 <- which(dataset$Outlet_Identifier == "OUT010")
dataset[index1, "Outlet_Size"] <- "Small"

index2 <- which(dataset$Outlet_Identifier == "OUT017")
dataset[index2, "Outlet_Size"] <- "Small"

index3 <- which(dataset$Outlet_Identifier == "OUT045")
dataset[index3, "Outlet_Size"] <- "Medium"
View(dataset)

summary(dataset)

---------------------------------#Visualization of Data-------------------------------------
#install.packages('Amelia')
library(Amelia)                 # for visualizing the missing data
missmap(traindata)       

# item Outlet Sales Histogram
ggplot(dataset, aes(x=Item_Outlet_Sales)) +
  geom_histogram(binwidth = 200) +
  labs(title = "Item Outlet Sales Histogram", 
       x = "Item Outlet Sales")

# item Outlet Sales Histogram by Outlet Identifier
ggplot(dataset, aes(x=Item_Outlet_Sales, 
                             fill = Outlet_Identifier)) +
  geom_histogram(binwidth = 100) +
  facet_wrap(~ Outlet_Identifier) +
  labs(title = "Item Outlet Sales Histogram", 
       x = "Item Outlet Sales")

# Sales by Outlet Identifier
ggplot(dataset, aes(x = Outlet_Identifier,
                             y = Item_Outlet_Sales)) +
  geom_boxplot() +
  labs(title = "Sales by Outlet Identifier",
       x = "Outlet Identifier",
       y = "Item Outlet Sales") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# item Outlet Sales by Item MRP and Outlet Identifier
ggplot(dataset aes(x = Item_MRP,
                             y = Item_Outlet_Sales)) +
  geom_bin2d() +
  facet_wrap(~ Outlet_Identifier) +
  labs(title = "Item Outlet Sales by Item MRP and Outlet Identifier",
       x = "Item MRP",
       y = "Item Outlet Sales")

#install.packages('GGally')
library(GGally)
as.data.frame(dataset)
class(dataset)
ggpairs(dataset$Outlet_Identifier) # scatter plot of outlet identifier with all other vars

# Median Sales by Location
dataset %>%
  group_by(Outlet_Identifier) %>%
  summarize(median_sales = median(Item_Outlet_Sales)) %>%
  arrange(desc(median_sales))

# Correlation of Item Outlet Sales and Item MRP
cor(dataset$Item_MRP, dataset$Item_Outlet_Sales)
cor(dataset$Item_Type, dataset$Item_Outlet_Sales)

# Preparing Data For Machine Learning
D1 <- dataset %>%
  select(-Item_Identifier, -Outlet_Identifier)
D1

# splitting the dataset
library(caTools)
set.seed(123)
split=sample.split(D1$Item_Outlet_Sales,SplitRatio = 0.7)
train=subset(D1,split==TRUE)
test=subset(D1,split==FALSE)

##OR (using Caret)
library(caret)
library(caretEnsemble)
set.seed(200)
trm <- createDataPartition(y = D1$Item_Outlet_Sales, 
                           p = 0.7, list=FALSE)
train <- D1[trm, ]
test<- D1[-trm, ]
table(complete.cases(D1$Item_Outlet_Sales))

View(train)

----------------------------------# MACHINE LEARNING MODELS-------------------------------

# Multiple Linear Regression

# finding optimal model and predictor variables via tweaking predictor variables
regressor=
  lm(formula=Item_Outlet_Sales~Item_Weight
     +Item_Visibility
     +Item_MRP+Outlet_Location_Type
     +Outlet_Type,data =dataset)
regressor
summary(regressor)$adj.r.squared

regressor1=
  lm(formula=Item_Outlet_Sales~Item_Weight
     +Item_Visibility
     +Item_MRP
     +Outlet_Type,data =dataset)
summary(regressor1)$adj.r.squared

regressor2=
  lm(formula=Item_Outlet_Sales~
     Item_Visibility
     +Item_MRP
     +Outlet_Type,data =dataset)
summary(regressor2)$adj.r.squared

regressor3=
  lm(formula=Item_Outlet_Sales~
     Item_MRP
     +Outlet_Type,data =dataset)
summary(regressor3)$adj.r.squared

regressorAll=
  lm(formula=Item_Outlet_Sales~.,data =dataset)
summary(regressorAll)$adj.r.squared

# Selected Linear model based on accuracy
model1=
  lm(formula=Item_Outlet_Sales~Item_Weight
     +Item_Visibility
     +Item_MRP+Outlet_Location_Type
     +Outlet_Type,data =train)

summary(model1)
summary(model1)$adj.r.squared
plot(model1)

# Detecting multicollinearity
library(MASS)
library(car)
library(psych)

regressor=
  lm(formula=Item_Outlet_Sales~.,data =train)
summary(regressor)$adj.r.squared

alias(regressor)
vif(regressor)   # variance inflation factor

# prepare training scheme and tuning parameters
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE)

# MODEL 1 : training the GLM model
modelGLM<-train(Item_Outlet_Sales ~Item_Weight
               +Item_Visibility
               +Item_MRP+Outlet_Location_Type
               +Outlet_Type,data=train,method = "glm",trControl = control)
modelGLM

# Testing Model
## Getting Predictions
pred_glm <- predict(modelGLM, test)
error <- pred_glm - test$Item_Outlet_Sales

#Calculating RMSE
sqrt(mean(error^2))

# training the GLMNET model
modelGLMNET<-train(Item_Outlet_Sales ~Item_Weight
               +Item_Visibility
               +Item_MRP+Outlet_Location_Type
               +Outlet_Type,data=train,method = "glmnet",na.action = na.fail,trControl = control)
modelGLMNET

# Testing Model
## Getting Predictions
pred_glmnet <- predict(modelGLMNET, test)
error <- pred_glmnet - test$Item_Outlet_Sales

## Calculating RMSE
sqrt(mean(error^2))

# MODEL 2: Random Forest and ranger(faster implementation of Random forest)
library(mlbench)
library(party)
library(randomForest)

forest<-randomForest(Item_Outlet_Sales~.,data=train,na.action=na.roughfix)
forest

modelRF<-train(Item_Outlet_Sales~Item_Weight
               +Item_Visibility
               +Item_MRP+Outlet_Location_Type
               +Outlet_Type,data=train,method  = "rf", preProcess = NULL,
               weights = NULL, metric = ifelse(is.factor(train$Item_Outlet_Sales), "Accuracy", "RMSE"),
               maximize = ifelse(metric %in% c("RMSE", "logLoss", "MAE"), FALSE,
                                 TRUE), na.action = na.roughfix,trControl = control)
modelRF

#OR

modelRF<-train(Item_Outlet_Sales~Item_Weight
               +Item_Visibility
               +Item_MRP+Outlet_Location_Type
               +Outlet_Type,data=train,method  = "ranger", na.action = na.roughfix,trControl = control)
modelRF

# Testing Model
## Getting Predictions
pred_rf <- predict(modelRF, test)
error <- pred_rf - test$Item_Outlet_Sales

## Calculating RMSE
sqrt(mean(error^2))

# MODEL 3: BagEarth (based on MARS: Multivariate adaptive regression spline)
#install.packages('earth')
library(earth)
bg<-bagEarth(Item_Outlet_Sales~Item_Weight
             +Item_Visibility
             +Item_MRP+Outlet_Location_Type
             +Outlet_Type,data=train,B=50,summary=mean,keepX=T)
bg

modelBagEarth<-train(Item_Outlet_Sales~Item_Weight
               +Item_Visibility
               +Item_MRP+Outlet_Location_Type
               +Outlet_Type,data=train,method  = "bagEarth", na.action = na.roughfix,trControl = control)
modelBagEarth

modelBagEarth1<-train(Item_Outlet_Sales~.,data=train,method  = "bagEarth", na.action = na.roughfix,trControl = control)
modelBagEarth1

# Testing Model
## Getting Predictions
pred_bg <- predict(modelBagEarth, test)
error <- pred_bg - test$Item_Outlet_Sales

## Calculating RMSE
sqrt(mean(error^2))

# MODEL 4: rpart ((Recursive Partitioning And Regression Trees)
#install.packages('party')
library(party)
ctreemodel<-ctree(Item_Outlet_Sales~Item_Weight
                  +Item_Visibility
                  +Item_MRP+Outlet_Location_Type
                  +Outlet_Type,data=train)
ctreemodel
plot(ctreemodel)

control<-trainControl(method='repeatedcv',number=10,repeats=3)

modelDTREE<-train(Item_Outlet_Sales~Item_Weight
                  +Item_Visibility
                  +Item_MRP+Outlet_Location_Type
                  +Outlet_Type,data=train,method='rpart',
                  parms=list(split='information'),trControl=control,tuneLength=10,na.action = na.roughfix)
modelDTREE

##install.packages('rpart.plot')
library(rpart.plot)
prp(modelDTREE$finalModel,box.palette='Red',tweak=1.4)

# Testing Model
## Getting Predictions
pred_dtree <- predict(modelDTREE, test)
error <- pred_dtree - test$Item_Outlet_Sales
## Calculating RMSE
sqrt(mean(error^2))

# MODEL 5: Stochastic Gradient Boosting (also known as Gradient Boosted Machine or GBM)
modelGBM<-train(Item_Outlet_Sales~.,data=train,method="gbm",trControl=control,na.action=na.roughfix)
modelGBM

# Testing Model
## Getting Predictions
pred_gbm <- predict(modelGBM, test)
error <- pred_gbm - test$Item_Outlet_Sales

## Calculating RMSE
sqrt(mean(error^2))

---------------------------------# TESTING MODELS ON TEST DATASET-----------------------------

testdata=read.csv('Test.csv') # reading test file
head(testdata)
View(testdata)

# Preprocessing test dataset
# Transforming "low fat" and "LF" to "Low Fat" and transforming "reg" to "Regular"
testdata$Item_Fat_Content=
  factor(testdata$Item_Fat_Content,
         levels =c('LF','Low Fat','low fat','Reg','Regular' ),
         labels = c('Low Fat','Low Fat','Low Fat','Regular','Regular'))

# testdata missing values imputation for Item_Weight (Replacing na values by mean)
testdata$Item_Weight=
  ifelse(is.na(testdata$Item_Weight),
         ave(testdata$Item_Weight,FUN=function(x)
           mean(x,na.rm = TRUE)),testdata$Item_Weight)

# Imputing the missing values for Outlet_Size (Replacing by Mode of Outlet_Size which is "Medium")
idx1 <- which(testdata$Outlet_Identifier == "OUT010")
testdata[idx1, "Outlet_Size"] <- "Small"

idx2 <- which(testdata$Outlet_Identifier == "OUT017")
testdata[idx2, "Outlet_Size"] <- "Small"

idx3 <- which(testdata$Outlet_Identifier == "OUT045")
testdata[idx3, "Outlet_Size"] <- "Medium"
View(testdata)

# Testing Predictions
testing_predictions<- predict(modelList, testdata)

testing_imputed$Item_Outlet_Sales <- testing_predictions_bag

submission_bag <- testing_imputed[, c("Item_Identifier",
                                      "Outlet_Identifier",
                                      "Item_Outlet_Sales")]

dim(submission_bag)

car::vif(modelGBM)
library(MASS)
library(car)
library(psych)
sqrt(vif(modelGLM))

# Model Performance
modelList <- c(modelGLM, modelGLMNET, modelRF, modelBagEarth, modelDTREE, modelGBM)
results <- resamples(modelList)
summary(results) # summarize the distributions

# boxplot of results
bwplot(results)
# dot plot of results
dotplot(results)
