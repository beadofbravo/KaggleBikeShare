##
## Bike Share EDA Code
##

## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(GGally)
library(tidymodels)

## Read in the Data
bike <- vroom("c://Users/peter/Desktop/BYU School/fall 2023/predictive analytics/KaggleBikeShare/train.csv")
# view(bike) to view data set
dplyr::glimpse(bike)
DataExplorer::plot_intro(bike)
DataExplorer::plot_correlation(bike) ## Shows high correlation between count and registered
                                     ## also between count and casual
DataExplorer::plot_bar(bike)
DataExplorer::plot_histogram(bike)
plot0 <- GGally::ggpairs(bike)

bike <- bike %>%
  mutate(weather = as.factor(weather))

dplyr::glimpse(bike)
## Initial plots

plot1 <- ggplot(data = bike, aes(x = atemp, y = count)) +
  geom_point() +
  geom_smooth(se = FALSE)

plot2 <- ggplot(data = bike, aes(x = humidity, y = count)) +
  geom_point() +
  geom_smooth(se = FALSE)

ggplot(data = bike, aes(x = windspeed, y = count)) +
  geom_point() +
  geom_smooth(se = FALSE)

plot3 <- ggplot(data = bike, aes(x = datetime, y = count)) +
  geom_point() +
  geom_smooth(se = FALSE)

ggplot(data = bike, aes(x = weather, y = count)) +
  geom_histogram() +
  geom_smooth(se = FALSE)

plot4 <- ggplot(data = bike) +
  geom_boxplot(aes(x = weather, y = count))

boxplot(x = bike$weather, y = bike$count)

## 4 panel layout
(plot1 + plot2) / (plot3 + plot4) #4 panel plot







