library(tidyverse)
library(tidymodels)
library(vroom)

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
  step_zv(all_predictors())
  ## change weather of 4 into 3
  


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

df %>%
  [,2] <- as.numeric([,2])
  pmax([,2], 0)

view(df)

?vroom_write
df <- data.frame(id = bike_test$datetime, bike_predict$.pred)
view(df)
vroom_write(x = df, file = "submission.csv")
