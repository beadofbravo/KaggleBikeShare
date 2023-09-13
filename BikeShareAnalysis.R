library(tidyverse)
library(tidymodels)
library(vroom)

## Read in the Data
bike <- vroom("./train.csv")

view(bike)
dplyr::glimpse(bike)

## Create a "recipe"
my_recipe <- recipe(count ~ atemp + season + weather + datetime + holiday + workingday + temp + humidity + windspeed + casual + registered, data = bike) %>%
  
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
  step_select(-c("registered", "casual"))


prepped_recipe <- prep(my_recipe) #set up processing using bike

bake(prepped_recipe, new_data=new_bike)

view(bike)
?factor


