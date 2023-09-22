library(tidyverse)
library(tidymodels)
library(vroom)
library(openxlsx)
library(lubridate)
library(poissonreg)
library(glmnet)
## Read in the Data
bike_test <- vroom("./test.csv")
bike_train <- vroom("./train.csv")
bike_train <- bike_train %>%
  select(-c('casual','registered'))


## Create a "recipe"
my_recipe <- recipe(count ~ ., data = bike_train) %>% 
  
  ## Feature Engineering Section
  ## make weather a factor
  step_mutate(weather=factor(weather)) %>%
  ## create hour and minutes variable
  step_time(datetime, features = c("hour", "minute")) %>%
  ## get days of the week
  step_date(datetime, features = "dow") %>%
  ## make season a factor
  step_mutate(season=factor(season)) %>%
  ## remove zero variance predictors
  step_zv(all_predictors()) %>%
  ## change weather of 4 into 3
  
  prep()


prep_train <- my_recipe %>%#set up processing using bike
  juice()

bake(my_recipe, new_data = bike_test)

dplyr::glimpse(prep_train)

view(prep_train)

## make night day variable
## make 4 weather into a three
## find more changes to do

my_mod <- linear_reg() %>% 
  set_engine("lm")

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train)

bike_predict <- predict(bike_workflow, new_data=bike_test)
view(bike_predict)

df <- data.frame(id = bike_test$datetime, bike_predict$.pred)
colnames(df) <- c("datetime", "count") # change column names to fit format
df_positive <- df
df_positive[df_positive < 0] <- 0
df_positive$datetime <- as.POSIXlt(df_positive$datetime)
df_positive$datetime <- format(as.POSIXct(df_positive$datetime,
                               format = "%m/%d/%Y %H:%M:%S"))
dplyr::glimpse(df_positive)
view(df_positive)

## looking at the fitted LM model
extract_fit_engine(bike_workflow) %>%
  summary()
extract_fit_engine(bike_workflow) %>%
  tidy()


vroom_write(x = df_positive, file = "./submission.csv", delim = ',')

## fitting a pois regression model

pois_mod <- poisson_reg() %>%
  set_engine("glm")

bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_train)

bike_pois_predictions <- predict(bike_pois_workflow, 
                                 new_data = bike_test)
view(bike_pois_predictions)
## Get Predictions for test set AND format for Kaggle
bike_pois_predictions <- bike_pois_predictions %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle


## Write prediction file to CSV
vroom_write(x=bike_pois_predictions, file="./submission2.csv", delim=",") 





## Penalized Regression

## load in bike_train to fix for penalized regression
bike_train_penreg <- vroom("./train.csv")

bike_train_penreg <- bike_train_penreg %>%
  select(-c('casual','registered')) %>%
  mutate(count = log(count))


my_recipe_pen <- recipe(count ~ ., data = bike_train_penreg) %>% 
  
  ## Feature Engineering Section
  ## make weather a factor
  step_mutate(weather=factor(weather)) %>%
  ## create hour and minutes variable
  step_time(datetime, features = c("hour", "minute")) %>%
  ## get days of the week
  step_date(datetime, features = "dow") %>%
  ## remove datetime
  step_rm(datetime) %>%
  ## make season a factor
  step_mutate(season=factor(season)) %>%
  ## remove zero variance predictors
  step_zv(all_predictors()) %>%
  ## change character variables to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  ## normalize numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  
  prep()


penreg_model <- linear_reg(penalty = .004, mixture=0) %>%
  set_engine("glmnet")

penreg_wf <- workflow() %>%
  add_recipe(my_recipe_pen) %>%
  add_model(penreg_model) %>%
  fit(data = bike_train_penreg)


## Get Predictions for test set AND format for Kaggle
log_lin_preds <- predict(penreg_wf, new_data = bike_test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write predictions to CSV
vroom_write(x=log_lin_preds, file="./PenReg&loglinPreds.csv", delim=",")



