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
  step_num2factor(weather, levels = c("Clear", "Few clouds", "Partly Cloudy", "Partly cloudy")) %>%
  ## create hour and minutes variable
  step_time(datetime, features = c("hour", "minute")) %>%
  ## get days of the week
  step_date(datetime, features = "dow") %>%
  ## make season a factor
  step_num2factor(season, levels = c("Spring", "Summer", "Fall", "Winter")) %>%
  ## remove zero variance predictors
  step_zv(all_predictors()) %>%
  
  ## Cleaning Section
  ## get rid the one day with weather = 4
  step_filter(weather != 4) %>%
  ## get rid of registered and casual columns, unusable in prediction
  prep()


prep_train <- my_recipe %>%#set up processing using bike
  juice()

bake(my_recipe, new_data = bike_test)

view(prep_train)
?factor


