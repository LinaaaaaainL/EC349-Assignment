### EC349 Project R Script
### Lina Tang u2144108
# load library [already connect with Github]
library(tidyverse)
library(caret)
library(hexbin) 
library(jsonlite)
library(magrittr)
library(dplyr)
library(glmnet)
library(ggplot2)
library(tree)
library(rpart)
library(rpart.plot)
install.packages("knitr")
library(knitr)
install.packages("tinytex")
tinytex::install_tinytex()
getwd()
setwd("C:/Users/123/Desktop/EC349 Assignment")

cat("\014")  
rm(list=ls())

# load data
load("C:/Users/123/Desktop/EC349 Assignment/yelp_review_small.Rda")
load("C:/Users/123/Desktop/EC349 Assignment/yelp_user_small.Rda")

# inspect data - review_data_small
head(review_data_small)
str(review_data_small)
summary(review_data_small)
dim(review_data_small)
colnames(review_data_small)

# inspect data - user_data_small
head(user_data_small)
str(user_data_small)
summary(user_data_small)
dim(user_data_small)
colnames(user_data_small)

# merge data by "user_id"
df <- review_data_small %>% left_join (user_data_small, by = "user_id")
str(df)
summary(df)
df$stars <- as.numeric(df$stars)

# deal with missing values - delete
sapply(df, function(x) sum(is.na(x)))
df = df[complete.cases(df), ]
str(df)
summary(df)

# set all main variables as numeric
df$useful.x <- as.numeric(df$useful.x)
df$funny.x <- as.numeric(df$funny.x)
df$cool.x <- as.numeric(df$cool.x)
df$funny.y <- as.numeric(df$funny.y)
df$useful.y <- as.numeric(df$useful.y)
df$fans <- as.numeric(df$fans)
df$average_stars <- as.numeric(df$average_stars)
df$compliment_cool <- as.numeric(df$compliment_cool)
df$compliment_more <- as.numeric(df$compliment_more)
df$compliment_hot <- as.numeric(df$compliment_hot)
df$compliment_profile <- as.numeric(df$compliment_profile)
df$compliment_note <- as.numeric(df$compliment_note)
df$compliment_cute <- as.numeric(df$compliment_cute)
df$compliment_list <- as.numeric(df$compliment_list)
df$compliment_funny <- as.numeric(df$compliment_funny)
df$compliment_plain <- as.numeric(df$compliment_plain)
df$compliment_writer <- as.numeric(df$compliment_writer)
df$compliment_photos <- as.numeric(df$compliment_photos)

# deal with outliers
hist(df$useful.x, breaks=10)
hist(df$funny.x, breaks=10)
hist(df$cool.x, breaks=10)
df <- subset(df, useful.x < 6 & funny.x < 6 & cool.x < 6)

# split data into training and test samples
set.seed(1)
df$stars <- as.numeric(df$stars)
training.samples <- df$stars %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data <- df[training.samples, ]
test.data <- df[-training.samples, ]



###### OLS Regression Analysis ######
# Build model 1
model1 <- lm(as.numeric(stars) ~ useful.x + funny.x + cool.x, data = train.data)
summary(model1)
plot(model1)

### . to avoid codes being 'eaten'
### .
### .
### .
# Prediction using model 1
predictions_m1 <- predict(model1, newdata = test.data)
compare <- data.frame(actual = as.numeric(test.data$stars), predicted = predictions_m1)
OLS_MSE_m1<- mean((predictions_m1 - as.numeric(test.data$stars)) ^ 2) #Note how it outperforms OLS
OLS_MSE_m1 #2.001981
sqrt(OLS_MSE_m1) #1.414914

# Model 2: with more variables
model2 <- lm(as.numeric(stars) ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_hot + compliment_more + compliment_profile + compliment_cute + compliment_list + compliment_note + compliment_plain + compliment_cool + compliment_funny + compliment_writer + compliment_photos, data = train.data)
summary(model2)
plot(model2)

### . to avoid codes being 'eaten'
### .
### .
### .
# Prediction using model 2
predictions_m2 <- model2 %>% predict(test.data)
compare <- data.frame(actual = as.numeric(test.data$stars),
                      predicted = predictions_m2)
OLS_MSE_m2<- mean((predictions_m2 - as.numeric(test.data$stars)) ^ 2) #Note how it outperforms OLS
OLS_MSE_m2 #1.332553
sqrt(OLS_MSE_m2) #1.154363

# Model 3: exclude insignificant affected variables
model3 <- lm(as.numeric(stars) ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_profile + compliment_more + compliment_list + compliment_writer + compliment_plain + compliment_note, data = train.data)
summary(model3)
plot(model3)

### . to avoid codes being 'eaten'
### .
### .
### .
# Prediction using model 3
predictions_m3 <- model3 %>% predict(test.data)
compare <- data.frame(actual = as.numeric(test.data$stars),
                      predicted = predictions_m3)
OLS_MSE_m3<- mean((predictions_m3 - as.numeric(test.data$stars)) ^ 2) #Note how it outperforms OLS
OLS_MSE_m3 #1.332569
sqrt(OLS_MSE_m3) #1.15437

# For Robust Standard Errors
library(sandwich)
library(lmtest)
coeftest(model3, vcov = vcovHC(model3, type="HC3"))



###### Ridge Regression ######
# Ridge on Train Dataset
library(glmnet)
library(ggplot2)

set.seed(1)
train_y <- train.data$stars
train_x <- data.matrix(train.data[, c('useful.x', 'funny.x', 'cool.x', 'useful.y', 'funny.y','cool.y', 'average_stars', 'compliment_hot', 'review_count', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos')])
test_y <- test.data$stars
test_x <- data.matrix(test.data[, c('useful.x', 'funny.x', 'cool.x', 'useful.y', 'funny.y','cool.y', 'average_stars', 'compliment_hot', 'review_count', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos')])

ridge.mod <- glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 0, lambda = 3, thresh = 1e-12)
summary(ridge.mod)

# perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 0, nfolds = 3)

# find optimal lambda value that minimizes test MSE
lambda_ridge_cv <- cv_model$lambda.min
lambda_ridge_cv #0.08563297

# find optimal lambda value that minimizes test MSE
plot(cv_model)

# find coefficients of best model
best_model <- glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 0, lambda = lambda_ridge_cv)
coef(best_model)

# use fitted best model to make predictions
y_predicted <- predict(ridge.mod, s = lambda_ridge_cv, newx = test_x)

# find SST and SSE
sst <- sum((test_y - mean(test_y))^2)
sse <- sum((y_predicted - test_y)^2)

# find R-Squared
rsq <- 1 - sse/sst
rsq #0.2098919

# calculateing RMSE for ridge model
ridge_MSE<- mean((y_predicted - test_y) ^ 2) #Note how it outperforms OLS
ridge_MSE #1.711681
sqrt(ridge_MSE) #1.308312



###### LASSO Analysis ######
cv.out2 <- cv.glmnet(as.matrix(train_x), as.matrix(train_y), alpha = 1, nfolds = 3)
plot(cv.out2)
lambda_LASSO_cv <- cv.out2$lambda.min #cross-validation is the lambda minimising empirical MSE in training data

#Re-Estimate Ridge with lambda chosen by Cross validation
LASSO.mod<-glmnet(train_x, train_y, alpha = 1, lambda = lambda_LASSO_cv, thresh = 1e-12)
coef(LASSO.mod) #note that some parameter estimates are set to 0 --> Model selection!

#Fit on Test Data
LASSO.pred <- predict(LASSO.mod, s = lambda_LASSO_cv, newx = as.matrix(test_x))
LASSO_MSE<- mean((LASSO.pred - test_y) ^ 2) #Note how it outperforms OLS
LASSO_MSE #1.332983
sqrt(LASSO_MSE) #1.154549



###### Decision Tree ######
#What happens when you include all variables?
mod1 <- glm(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_hot + compliment_more + compliment_profile + compliment_cute + compliment_list + compliment_note + compliment_plain + compliment_cool + compliment_funny + compliment_writer + compliment_photos, data = train.data)

#Review the results
coef(mod1) 
summary(mod1)

#Predicted Values for Test Data based on the model estimates
mod1_predict<-predict(mod1, newdata = test.data)

#What happens when you exclude insignificant variables?
mod2 <- glm(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data)

#Review the results
coef(mod2) 
summary(mod2)

# Classification Tree
library(tree)
set.seed(1)
train.data$stars <- as.factor(train.data$stars)
test.data$stars <- as.factor(test.data$stars)
train.data$useful.x <- as.factor(train.data$useful.x)
test.data$useful.x <- as.factor(test.data$useful.x)
train.data$funny.x <- as.factor(train.data$funny.x)
test.data$funny.x <- as.factor(test.data$funny.x)
train.data$cool.x <- as.factor(train.data$cool.x)
test.data$cool.x <- as.factor(test.data$cool.x)

#With tree library
tree1<-tree(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data)

#partition graph
partition.tree(tree1) 
points(train.data[, c("useful.x", "funny.x", "cool.x", "useful.y", "funny.y",   "cool.y", "average_stars", "fans", "review_count", "compliment_more",  "compliment_profile", "compliment_list", "compliment_note", "compliment_plain", "compliment_writer")], cex=.4)

plot(tree1)
text(tree1, pretty = 0)
title(main = "Unpruned Classification Tree")

# Classification Tree without restrictions
library(rpart)
rpart_tree<-rpart(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data)
rpart.plot(rpart_tree)

# Classification Tree with restrictions
rpart_tree2<-rpart(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data, control = rpart.control(maxdepth = 10, minsplit = 5, minbucket = 10000, cp = 0))
print(rpart_tree2)
rpart.plot(rpart_tree2, cex = 0.5)

#importance
rpart_tree2$variable.importance



###### Bagging ######
library(ipred)      
set.seed(1)      

#fit the bagged model
bag <- bagging(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data, nbagg = 100,   
               coob = TRUE, control = rpart.control(minsplit = 3, cp = 0.1)
)

#display fitted bagged model
bag



###### Random Forests ######
library(randomForest) #randomforest
set.seed(1)
model_RF<-randomForest(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data, ntree=200)
pred_RF_test = predict(model_RF, test.data)
mean(model_RF[["err.rate"]]) #0.5907672 for 200 trees.


###### Boosting ######
library(adabag) #AdaBoost

# train a model using our training data
model_adaboost <- boosting(stars ~ useful.x + funny.x + cool.x + useful.y + funny.y + cool.y + average_stars + fans + review_count + compliment_more + compliment_profile + compliment_list + compliment_note + compliment_plain + compliment_writer, data = train.data, boos=TRUE, mfinal=50)
summary(model_adaboost)

# Load data for Regression (this one is stored on R)
#use model to make predictions on test data
pred_ada_test = predict(model_adaboost, test.data)

# Returns the prediction values of test data along with the confusion matrix
pred_ada_test

