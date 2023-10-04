####################
## Random Forests ##
####################
library(rpart)
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(ranger)
## Read in the Data
bike_train <- vroom("./train.csv")
bike_test <- vroom("./test.csv")
## Data Cleaning

bike_train <- bike_train %>%
  select(-casual,-registered)
bike_train_rf <- bike_train %>%
  mutate(count=log(count))

### Random Forest
RF_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) %>% #Type of model (500 or 1000)
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")


## Create a workflow with model & recipe


my_recipe_rf <- recipe(count ~ ., data = bike_train_rf) %>% 
  
  ## Feature Engineering Section
  ## make weather a factor
  step_mutate(weather=factor(weather)) %>%
  ## make year variable
  step_date(datetime, features = c("year")) %>%
  ## make year a factor
  step_mutate(datetime_year = factor(datetime_year)) %>%
  ## create hour and minutes variable
  step_time(datetime, features = c("hour")) %>%
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
  
  prep()

RF_wf <- workflow() %>%
  add_recipe(my_recipe_rf) %>%
  add_model(RF_mod)


## Set up grid of tuning values
RF_tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                               min_n())


## Set up K-fold CV
folds <- vfold_cv(bike_train_rf, v = 5, repeats=1)


## Run the CV
CV_results_RF <- RF_wf %>%
  tune_grid(resamples=folds,
            grid=RF_tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL


## Find best tuning parameters
bestTune <- CV_results_RF %>%
  select_best("rmse")


## Finalize workflow and predict
final_wf <-
  RF_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bike_train_rf)


## Predict
final_wf %>%
  predict(new_data = bike_test)


test_preds <- final_wf %>%
  predict(new_data = bike_test) %>%
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write prediction file to CSV
vroom_write(x=test_preds, file="./RFSubmission.csv", delim=",")
