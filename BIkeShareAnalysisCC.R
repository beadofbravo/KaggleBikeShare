library(tidyverse)
library(tidymodels)
library(vroom)
library(openxlsx)
library(lubridate)
library(poissonreg)
library(glmnet)

bike_train <- vroom("./train.csv")
bike_train_penreg <- bike_train %>%
  select(-c('casual','registered')) %>%
  mutate(count = log(count))
bike_test <- vroom("./test.csv")

my_recipe_pen <- recipe(count ~ ., data = bike_train_penreg) %>% 
  
  ## Feature Engineering Section
  ## make weather a factor
  step_mutate(weather=factor(weather)) %>%
  ## create hour and minutes variable
  step_time(datetime, features = c("hour", "minute")) %>%
  ## get days of the week
  step_date(datetime, features = "dow") %>%
  ## make weekend variable for FRI, SAT, SUN
  step_mutate(weekend = case_when(datetime_dow == "Fri" ~ 1,
                                  datetime_dow == "Sat" ~ 1,
                                  datetime_dow == "Sun" ~ 1,
                                  TRUE ~ 0)) %>%
  ## remove datetime
  step_rm(datetime) %>%
  ## make season a factor
  step_mutate(season=factor(season)) %>%
  ## make hours a factor
  step_mutate(datetime_hour = factor(datetime_hour)) %>%
  ## remove zero variance predictors
  step_zv(all_predictors()) %>%
  ## change character variables to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  ## normalize numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  
  prep()

prep_train <- my_recipe_pen %>%#set up processing using bike
  juice()

## set up the model
penreg_model <- linear_reg(penalty = 0, mixture=1) %>%
  set_engine("glmnet")

## workflow for penreg
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
vroom_write(x=log_lin_preds, file="./inclasschallenge.csv", delim=",")













