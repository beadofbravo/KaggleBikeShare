library(tidyverse)
library(tidymodels)
library(vroom)
library(openxlsx)
library(lubridate)
library(poissonreg)
library(glmnet)
library(rpart)
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

####################################
## fitting a pois regression model##
####################################
pois_mod <- poisson_reg() %>%
  set_engine("glm")

bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_train)

bike_pois_predictions <- predict(bike_pois_workflow, 
                                 new_data = bike_test)

## Get Predictions for test set AND format for Kaggle
bike_pois_predictions <- bike_pois_predictions %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle


## Write prediction file to CSV
vroom_write(x=bike_pois_predictions, file="./submission2.csv", delim=",") 




##########################
## Penalized Regression ##
##########################

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


###################################
## cross validation tuning stuff ##
###################################


## setting up data set
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
  ## remove zero variance predictors
  step_zv(all_predictors()) %>%
  ## change character variables to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  ## normalize numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  
  prep()


## Penalized Reg Model

pen_reg_val_model <- linear_reg(penalty = tune(),
                                mixture = tune()) %>%
  set_engine("glmnet")


## Workflow
pen_reg_val_wf <- workflow() %>%
  add_recipe(my_recipe_pen) %>%
  add_model(pen_reg_val_model)


## grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)


## Split training data for cross validation
folds <- vfold_cv(bike_train_penreg, v = 5, repeats = 1)


## Run the cross validation
CV_results <- pen_reg_val_wf %>%
  tune_grid(resamples = folds, 
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))


## Plot of results
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) + 
  geom_line()


## Finding the best values
bestTune <- CV_results %>%
  select_best("rmse")


## Final Workflow
final_wf <- pen_reg_val_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train_penreg)


## Predictions
cv_preds <- final_wf %>%
  predict(new_data = bike_test)

view(cv_preds)


## Get Predictions for test set AND format for Kaggle for cross validation
cv_preds <- cv_preds %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write predictions to CSV
vroom_write(x=cv_preds, file="./cv_preds.csv", delim=",")




######################
## Regression Trees ##
######################

## read in data
bike_train <- vroom("./train.csv")
bike_train_logcount <- bike_train %>%
  select(-c('casual','registered')) %>%
  mutate(count = log(count))
bike_test <- vroom("./test.csv")


## Recipe

my_recipe_regtree <- recipe(count ~ ., data = bike_train_logcount) %>% 
  
  ## Feature Engineering Section
  ## make weather a factor
  step_mutate(weather=factor(weather)) %>%
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


## set up the model for regression trees
my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")


## workflow

pen_reg_tree_wf <- workflow() %>%
  add_recipe(my_recipe_regtree) %>%
  add_model(my_mod)

## setting up tuning values

tuning_grid_regtree <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n())


## Split training data for cross validation
folds <- vfold_cv(bike_train_logcount, v = 5, repeats = 1)


## Run the cross validation
CV_results_regtree <- pen_reg_tree_wf %>%
  tune_grid(resamples = folds, 
            grid = tuning_grid_regtree,
            metrics = metric_set(rmse, mae, rsq))


## Finding the best values
bestTune <- CV_results_regtree %>%
  select_best("rmse")


## Final Workflow
final_wf_regtree <- pen_reg_tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train_logcount)


## Predictions
cv_preds_regtree <- final_wf_regtree %>%
  predict(new_data = bike_test)


## Get Predictions for test set AND format for Kaggle for cross validation
cv_preds_regtree <- cv_preds_regtree %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write predictions to CSV
vroom_write(x=cv_preds_regtree, file="./regtree_preds.csv", delim=",")



####################
## Random Forests ##
####################







####################
## Model Stacking ##
####################
library(tidymodels)
library(tidyverse)
library(stacks)
library(vroom)

bike_train <- vroom("./train.csv")
bike_train_modstack <- bike_train %>%
  select(-c('casual','registered')) %>%
  mutate(count = log(count))
bike_test <- vroom("./test.csv")

my_recipe_modstack <- recipe(count ~ ., data = bike_train_modstack) %>% 
  
  ## Feature Engineering Section
  ## make weather a factor
  step_mutate(weather=factor(weather)) %>%
  ## change weather 4 into weather 3
  step_mutate(weather = ifelse(weather == 4, 3, weather))%>%
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
  ## remove zero variance predictors
  step_zv(all_predictors()) %>%
  ## change character variables to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  ## normalize numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  
  prep()

## cross-validation folds
folds <- vfold_cv(bike_train_modstack, v = 10)

## control settings for stacking models
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()


## Set up linear model

lin_model_modstack <- linear_reg() %>%
  set_engine("lm")

## Set workflow
linreg_wf_modstack <- workflow() %>%
  add_recipe(my_recipe_modstack) %>%
  add_model(lin_model_modstack)

## fit linear to folds
linreg_folds_fit <- linreg_wf_modstack %>%
  fit_resamples(resamples = folds,
                control = tunedModel)


## Penalized Regression

## define the model
pen_reg_model_modstack <- linear_reg(mixture = tune(),
                                     penalty = tune()) %>%
  set_engine("glmnet")


## define a workflow
pen_reg_wf_modstack <- workflow() %>%
  add_recipe(my_recipe_modstack) %>%
  add_model(pen_reg_model_modstack)

## define grid of tuning values
pen_reg_tune_grid <- grid_regular(mixture(),
                                  penalty(),
                                  levels = 7)

## fit into folds
pen_reg_fold_fit <- pen_reg_wf_modstack %>%
  tune_grid(resamples = folds,
            grid = pen_reg_tune_grid,
            metrics = metric_set(rmse),
            control = untunedModel)


## Reg Tree

## set up the model for regression trees
regtree_modstack <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # Engine = What R function to use
  set_mode("regression")


## Workflow
regTree_wf_modstack <- workflow() %>%
  add_recipe(my_recipe_modstack) %>%
  add_model(regtree_modstack)


## Grid for tuning

regtree_modstack_tunegrid <- grid_regular(tree_depth(),
                                          cost_complexity(),
                                          min_n(),
                                          levels = 5)

## Tune the Model
tree_folds_fit_modstack <- regTree_wf_modstack %>%
  tune_grid(resamples = folds,
            grid = regtree_modstack_tunegrid,
            metrics = metric_set(rmse),
            control = untunedModel)


## Stacking time

bike_stack <- stacks() %>%
  add_candidates(linreg_folds_fit) %>%
  add_candidates(pen_reg_fold_fit) %>%
  add_candidates(tree_folds_fit_modstack)

fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()


## Predictions
modstack_preds <- predict(fitted_bike_stack, new_data = bike_test)



## Get Predictions for test set AND format for Kaggle for cross validation
modstack_preds <- modstack_preds %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write predictions to CSV
vroom_write(x=modstack_preds, file="./modstack_preds.csv", delim=",")

view(modstack_preds)





