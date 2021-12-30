# Multiple approaches to predict flake mass

-   Guillermo Bustos-Pérez <sup>(1,2)</sup>  
-   Javier Baena Preysler <sup>(1)</sup>

<sup>1</sup> Departamento de Prehistoria y Arqueología, Universidad
Autónoma de Madrid, Madrid, Spain  
<sup>2</sup> Corresponding author at:
<guillermo.bustos@estudiante.uam.es> \| <guillermo.willbustos@gmail.com>

## Table of contents

-   01 Installing packages  
-   02 Loading and describing the data
    -   02.1 Loading and describing the data  
    -   02.2 Variable selection  
-   03 Model training and hyperparameter tunning:
    -   03.1 Multiple Linear Regression  
    -   03.2 Random Forest Regression  
    -   03.3 Artificial Neuronal Network (ANN)  
-   04 Model evaluation
    -   04.1 Model evaluation metrics  
    -   04.2 Variable importance

## 01 Installing packages

The following code provides the list of packages employed in the
analysis, checks if they are missing and installs the missing ones. This
is set to meet reproducibility standards for machine learning (Heil et
al., 2021).

``` r
list.of.packages <- c("tidyverse", "caret", "neuralnet", "lattice", "ranger")

new.packages <- list.of.packages[!(list.of.packages %in% 
                                     installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages)
```

 

After this we can load the packages to perform model training and
analysis. Additionally in this markdown we are going to use package
knitr to show a nice output of tables.

``` r
list.of.packages <- c("tidyverse", "caret",  "ranger", "knitr", "knitr")

lapply(list.of.packages, library, character.only = TRUE)
```

    ## [[1]]
    ##  [1] "forcats"   "stringr"   "dplyr"     "purrr"     "readr"     "tidyr"    
    ##  [7] "tibble"    "ggplot2"   "tidyverse" "stats"     "graphics"  "grDevices"
    ## [13] "utils"     "datasets"  "methods"   "base"     
    ## 
    ## [[2]]
    ##  [1] "caret"     "lattice"   "forcats"   "stringr"   "dplyr"     "purrr"    
    ##  [7] "readr"     "tidyr"     "tibble"    "ggplot2"   "tidyverse" "stats"    
    ## [13] "graphics"  "grDevices" "utils"     "datasets"  "methods"   "base"     
    ## 
    ## [[3]]
    ##  [1] "ranger"    "caret"     "lattice"   "forcats"   "stringr"   "dplyr"    
    ##  [7] "purrr"     "readr"     "tidyr"     "tibble"    "ggplot2"   "tidyverse"
    ## [13] "stats"     "graphics"  "grDevices" "utils"     "datasets"  "methods"  
    ## [19] "base"     
    ## 
    ## [[4]]
    ##  [1] "knitr"     "ranger"    "caret"     "lattice"   "forcats"   "stringr"  
    ##  [7] "dplyr"     "purrr"     "readr"     "tidyr"     "tibble"    "ggplot2"  
    ## [13] "tidyverse" "stats"     "graphics"  "grDevices" "utils"     "datasets" 
    ## [19] "methods"   "base"     
    ## 
    ## [[5]]
    ##  [1] "knitr"     "ranger"    "caret"     "lattice"   "forcats"   "stringr"  
    ##  [7] "dplyr"     "purrr"     "readr"     "tidyr"     "tibble"    "ggplot2"  
    ## [13] "tidyverse" "stats"     "graphics"  "grDevices" "utils"     "datasets" 
    ## [19] "methods"   "base"

 

## 02 Loading and describing the data

Sample for analysis is composed of 500 experimentally knapped flakes
using hard hammer. Flakes belong to 30 knapping sequences where a wide
variety of knapping methods were employed —hierarchical (Levallois and
Hierarchical Discoid), bifacial (Discoid), and unipolar— to generate the
experimental sample, ensuring a wide range of morphologies. This is an
expansion of a previous dataset employed for similar purposes
(Bustos-Pérez and Baena, 2021) which allows to expand the range of
dimensions and mass of the assemblage.

``` r
# Load the data
Reg_Data <- read.csv("Data/Flake Mass v02 Eng.csv")
```

``` r
kable(Reg_Data[1:10,])
```

| Length | Width | MeanThick | Max_Thick | Weight | Surface.Plat | Platfom_Depth | Cortex | No_Scars | Termination_type | EPA | Log_Weight | Log_Max_Thick | Log_Plat | Log_Plat_De |
|-------:|------:|----------:|----------:|-------:|-------------:|--------------:|-------:|---------:|:-----------------|----:|-----------:|--------------:|---------:|------------:|
|   51.3 |  29.8 | 10.066667 |      13.1 |  17.83 |       83.585 |           7.3 |      5 |        4 | Feather          |  51 |  1.2511513 |     1.1172713 | 1.922128 |   0.8633229 |
|   49.1 |  30.0 |  8.566667 |       9.7 |  13.33 |       90.480 |           7.8 |      5 |        3 | Feather          |  70 |  1.1248301 |     0.9867717 | 1.956553 |   0.8920946 |
|   30.8 |  43.8 | 11.566667 |      16.8 |  20.33 |       40.500 |           3.6 |      3 |        2 | Feather          |  35 |  1.3081374 |     1.2253093 | 1.607455 |   0.5563025 |
|   30.2 |  19.6 |  5.500000 |       6.7 |   3.98 |       59.670 |           5.1 |      5 |        3 | Feather          |  66 |  0.5998831 |     0.8260748 | 1.775756 |   0.7075702 |
|   57.1 |  37.8 | 11.166667 |      13.3 |  22.18 |      109.800 |          12.0 |      4 |        3 | Feather          |  68 |  1.3459615 |     1.1238516 | 2.040602 |   1.0791812 |
|   37.5 |  34.2 |  5.466667 |       6.7 |   7.97 |       51.340 |           6.8 |      5 |        1 | Hinge            |  65 |  0.9014583 |     0.8260748 | 1.710456 |   0.8325089 |
|   65.6 |  41.9 | 10.400000 |      14.6 |  24.16 |       93.840 |          10.2 |      5 |        2 | Feather          |  67 |  1.3830969 |     1.1643529 | 1.972388 |   1.0086002 |
|   86.8 |  70.8 | 16.066667 |      19.2 |  96.20 |      210.625 |          12.5 |      5 |        3 | Step             |  66 |  1.9831751 |     1.2833012 | 2.323510 |   1.0969100 |
|   39.2 |  54.7 | 16.700000 |      27.3 |  31.70 |       17.460 |           3.6 |      5 |        2 | Feather          |  30 |  1.5010593 |     1.4361626 | 1.242044 |   0.5563025 |
|   49.2 |  60.6 | 11.233333 |      14.0 |  40.16 |      158.080 |          12.8 |      5 |        2 | Hinge            |  68 |  1.6037937 |     1.1461280 | 2.198877 |   1.1072100 |

``` r
#  Summary statistics of the experimental assemblage
Summary_Assem <- data.frame(
  rbind(data.frame(data.matrix(summary(Reg_Data$Length))) %>% t(),
        data.frame(data.matrix(summary(Reg_Data$Width))) %>% t(),
        data.frame(data.matrix(summary(Reg_Data$MeanThick))) %>% t(),
        data.frame(data.matrix(summary(Reg_Data$Surface.Plat))) %>% t(),
        data.frame(data.matrix(summary(Reg_Data$Weight))) %>% t()))
Measure <- c("Length", "Width", "Mean Thickness", "Platform Surface",
             "Weight")
Summary_Assem <- cbind(Measure, Summary_Assem)
rownames(Summary_Assem) <- 1:nrow(Summary_Assem)
```

``` r
kable(Summary_Assem)
```

| Measure          |      Min. |  X1st.Qu. |    Median |      Mean |  X3rd.Qu. |   Max. |
|:-----------------|----------:|----------:|----------:|----------:|----------:|-------:|
| Length           | 16.500000 | 36.300000 | 45.900000 | 48.253200 |  59.60000 | 100.90 |
| Width            | 14.900000 | 31.175000 | 39.000000 | 40.559200 |  46.82500 |  85.50 |
| Mean Thickness   |  1.800000 |  6.058333 |  8.516667 |  9.249567 |  11.28333 |  26.50 |
| Platform Surface |  2.591814 | 31.350000 | 62.933736 | 93.254685 | 116.11875 | 620.00 |
| Weight           |  1.140000 |  5.870000 | 12.965000 | 21.390400 |  26.95750 | 200.73 |

``` r
kable(data.frame(table(Reg_Data$Termination_type)))
```

| Var1     | Freq |
|:---------|-----:|
| Feather  |  449 |
| Hinge    |   42 |
| Inflexed |    2 |
| Plunging |    2 |
| Step     |    5 |

 

A fast to explore lithic assemblage composition is through a Bagolini
scatter plot (Bagolini, 1968). Comparison of the experimental dataset
with the one of the previous study (Bustos-Pérez and Baena, 2021) shows
an increase on the size and average mass of experimentally knapped
flakes. While in the previous study 50% of the flakes had mass values
between 4.15g and 14.02g (Bustos-Pérez and Baena, 2021), in the present
study 50% of the flakes weight between 5.87g and 26.96g. This indicates
that the expansion of the dataset has been done by the inclusion of
heavier and bigger flakes.

``` r
Reg_Data %>% 
  ggplot(aes(Width, Length)) +
  geom_segment(x = 40, y = 0, xend = 0, yend = 40, color = "gray48") +
  geom_segment(x = 60, y = 0, xend = 0, yend = 60, color = "gray48") +
  geom_segment(x = 80, y = 0, xend = 0, yend = 80, color = "gray48") +
  
  geom_segment(x = 0, y = 0, xend = 105, yend = 105, color = "gray48") +
  
  geom_segment(x = 0, y = 0, xend = (105/6), yend = 105, color = "gray48") +
  geom_segment(x = 0, y = 0, xend = (105/3), yend = 105, color = "gray48") +
  geom_segment(x = 0, y = 0, xend = (105/2), yend = 105, color = "gray48") +
  geom_segment(x = 0, y = 0, xend = (105/1.5), yend = 105, color = "gray48") +
  geom_segment(x = 0, y = 0, xend = (105/0.75), yend = 105, color = "gray48") +
  geom_segment(x = 0, y = 0, xend = (105/0.5), yend = 105, color = "gray48") +
  geom_segment(x = 0, y = 0, xend = 105, yend = (105/2), color = "gray48") +
  
  annotate("text", x = 0, y = 104, adj = 0, 
           label = "Very thin blade", size = 2.5) +
  annotate("text", x = 20, y = 104, adj = 0, 
           label = "Thin blade", size = 2.5) +
  annotate("text", x = 40, y = 104, adj = 0, 
           label = "Blade", size = 2.5) +
  annotate("text", x = 53, y = 104, adj = 0, 
           label = "Elongated flake", size = 2.5) +
  annotate("text", x = 85, y = 104, adj = 0, 
           label = "Flake", size = 2.5) +
  annotate("text", x = 103, y = 92.5, adj = 0, 
           label = "Wide\nflake", size = 2.5) +
  annotate("text", x = 103, y = 65, adj = 0, 
           label = "Very\nwide\nflake", size = 2.5) +
  annotate("text", x = 103, y = 25, adj = 0, 
           label = "Wider\nflake", size = 2.5) +
  
  annotate("text", x = 20, y = 1, adj = 0, 
           label = "Micro", size = 2.5) +
  annotate("text", x = 47, y = 1, adj = 0, 
           label = "Small", size = 2.5) +
  annotate("text", x = 65, y = 1, adj = 0, 
           label = "Normal", size = 2.5) +
  annotate("text", x = 85, y = 1, adj = 0, 
           label = "Big", size = 2.5) +
  
  geom_point(aes(color = Termination_type), size = 2, alpha = 0.75) +
  scale_x_continuous(breaks = seq(0, 105, 5), lim = c(0, 105)) +
  scale_y_continuous(breaks = seq(0, 105, 5), lim = c(0, 105)) +
  ylab("Length (mm)") +
  xlab("Width (mm)") +
  theme_light() +
  ggsci::scale_color_aaas() +
  labs(color = "Termination type") +
  guides(color = guide_legend(nrow = 1, title.position = "top")) +
  theme(axis.title = element_text(size = 9, color = "black", face = "bold"),
        axis.text = element_text(size = 7.5, color = "black"),
        legend.position = "bottom") +
  coord_fixed() 
```

![](01-Complete-script_files/figure-markdown_github/Baggolini%20scatter%20plot-1.png)
 

Additionally, exploratory visual analysis of flake mass through
histogram shows a highly skewed distribution with flakes weighting
between 10 g and 20 g the most frequent.

``` r
# Histogram of flake weight
Reg_Data %>% ggplot(aes(Weight)) +
  geom_histogram(binwidth = 10,
                 color = "black", fill = "gray") +
  theme_light() +
  ylab("Count") +
  xlab("Weight (g)") +
  scale_x_continuous(breaks = seq(0, 200, 20)) +
  theme(
    axis.text = element_text(color = "black", size = 9),
    axis.title = element_text(color = "black", size = 10))
```

![](01-Complete-script_files/figure-markdown_github/Histogramm%20of%20flake%20weight-1.png)
 

## 03 Model training

### 03.1 Multiple Linear regression

Multiple linear regression extends the simple linear regression to
accomodate multiple predictors

``` r
### Set Train control ####
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 50,
                              savePredictions = TRUE)

# Train the model
frmla <- Log_Weight ~ MeanThick + Cortex + No_Scars + EPA + Log_Max_Thick + Log_Plat + Log_Plat_De

set.seed(123)
lm.model <- train(frmla, 
               data = Reg_Data, 
               method = "lm",
               trControl = train.control)
```

``` r
summary(lm.model)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.69473 -0.13616  0.01939  0.13186  0.47859 
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   -0.4427689  0.1301456  -3.402 0.000723 ***
    ## MeanThick      0.0337557  0.0064369   5.244 2.34e-07 ***
    ## Cortex        -0.0814959  0.0093836  -8.685  < 2e-16 ***
    ## No_Scars       0.0731414  0.0101348   7.217 2.03e-12 ***
    ## EPA            0.0017785  0.0009083   1.958 0.050805 .  
    ## Log_Max_Thick  0.9273949  0.1430721   6.482 2.20e-10 ***
    ## Log_Plat       0.3358338  0.0469619   7.151 3.13e-12 ***
    ## Log_Plat_De   -0.3522401  0.0898045  -3.922 0.000100 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.2083 on 492 degrees of freedom
    ## Multiple R-squared:  0.779,  Adjusted R-squared:  0.7759 
    ## F-statistic: 247.8 on 7 and 492 DF,  p-value: < 2.2e-16

 

### 03.2 Random Forest Regression

Random Forests build multiple decision trees from the training data
(Breiman, 2001). Using different data adds diversity to the models.
Predictiobs are obtained through the average of examples that reach a
leaf. Finally, the results are averaged along all the grown trees.

Cartesian grid search is performed on the following hyperparameters:
number of trees to grow for each model (ranging from 500 to 700 by 25);
number of variables to possibly split at each node (ranging from 1 to
5); and minimal node size (ranging from 1 to 5).

Please note that while **mtry** (random number of variables to consider
in each split) and **min.node.size** can be integrated into a cartesian
grid search, the number of trees to grow cannot. Thus, it is necessary
to include the training of the models in a loop were the number of trees
changes in a sequence. At the end of each loop the best combination of
mtry and min.node.size for the given number of trees is extracted (along
with the results).

``` r
#### Hyperparameter tuning for random forest ####

# Train control
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 50,
                           verboseIter = TRUE)

# range of hyperparameters
mtry <- seq(1, 5, 1)
min.node.size <- seq(1, 5, 1)
splitrule = "variance"

# Grid of possible combinations
hyper_grid <- expand.grid(
  mtry = mtry,
  min.node.size = min.node.size ,
  splitrule = "variance")

# Loop over different number of trees
best_tune <- data.frame(
  mtry = numeric(0),
  min_node.size = numeric(0),
  Num_Trees = numeric(0),
  r_squared = numeric(0))

my_seq <- seq(500, 700, 25)

set.seed(123)
for (x in my_seq){
  
  RF_weight <- train(frmla, 
                       Reg_Data,
                       method = "ranger",
                       trControl = fitControl,
                       num.trees = x,
                       tuneGrid = hyper_grid 
  )
  
  Bst_R <- data.frame(
    mtry = RF_weight$bestTune[[1]],
    min_node.size = RF_weight$bestTune[[3]],
    Num_Trees = x,
    r_squared = RF_weight$finalModel[[10]])
  
  best_tune <- rbind(best_tune, Bst_R)
  
  Bst_R <- c()
  
}
```

 

The previous code is computationally expensive, but it ensures finding
the best combination of hyperparameters. The following table presents
the results of hyperparameter grid search.

``` r
kable(best_tune)
```

| mtry | min_node.size | Num_Trees | r_squared |
|-----:|--------------:|----------:|----------:|
|    2 |             5 |       500 | 0.7312673 |
|    2 |             5 |       525 | 0.7271587 |
|    2 |             5 |       550 | 0.7256057 |
|    2 |             5 |       575 | 0.7305429 |
|    2 |             4 |       600 | 0.7281003 |
|    2 |             4 |       625 | 0.7312961 |
|    2 |             5 |       650 | 0.7277806 |
|    2 |             5 |       675 | 0.7279673 |
|    2 |             4 |       700 | 0.7296423 |

On the table we can see that **mtry** is constant at a value of 2 for
all combinations of number of trees and minimum node size. Additionally
values of minimum node size range either from 4 or 5. This simplifies
the visualization of hyperparameters.

``` r
#### Hyperparameters of Random forest ####
data.frame(best_tune) %>% 
  ggplot(aes(factor(Num_Trees), min_node.size, fill = r_squared)) + 
  geom_tile(alpha = 0.75) +
  xlab("Number of trees") +
  ylab("Min node size") +
  scale_y_continuous(breaks = seq(4, 5, 1), lim = c(3.5, 5.5)) +
  geom_text(aes(label = round(r_squared, 4)), size = 3) +
  ggsci::scale_fill_gsea(reverse = TRUE) +
  theme_classic() +
  theme(legend.position = "none",
        axis.text = element_text(color = "black", size = 8))
```

![](01-Complete-script_files/figure-markdown_github/graph%20of%20random%20forest%20hyperparamters-1.png)

 

The following code returns the best combination of hyperparameters.

``` r
best_tune_2 <- best_tune[which.max(best_tune$r_squared),]
best_tune_2
```

    ##   mtry min_node.size Num_Trees r_squared
    ## 6    2             4       625 0.7312961

 

Finally, the Random forest with optimal combination of hyperparameters
can be trained.

``` r
#### Final random forest model ####
newr_grid <- expand.grid(mtry = best_tune_2$mtry,
                          min.node.size = best_tune_2$min_node.size,
                          splitrule = "variance"
)

# Train control
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 50)

RF_weight <- train(frmla, 
                   Reg_Data,
                   method = "ranger",
                   trControl = fitControl,
                   tuneGrid = newr_grid, 
                   num.trees = best_tune$Num_Trees,
                   importance = "impurity_corrected")
```

 

### 03.3 Artificial Neuronal Network (ANN)

Artificial Neuronal Networks (ANN) model the relationship between input
data and the output signal through a series of hidden layers each
composed by a number of nodes (Lantz, 2015). The present work uses the R
package “neuralnet” (Günther and Fritsch, 2010) to train ANN with
backpropagation (Rumelhart et al., 1986). For the present work ANN
topology is limited to having only one or two hidden layers. Number of
nodes of hidden layer 1 ranges between 1 and 4 while number of nodes of
hidden layer 2 ranges from 0 (no second hidden layer) to 4. All possible
combination are tested.

``` r
#### Look for best ANN architecture ####
 set.seed(123)
 train.control <- trainControl(method = "repeatedcv", 
                               number = 10, repeats = 50,
                               verboseIter = TRUE)

  tune.grid.neuralnet <- expand.grid(
   .layer1 = c(1:4),
   .layer2 = c(0:4),
   .layer3 = 0
 )

 nnet_model <- train(
   Log_Weight ~ MeanThick + Log_Max_Thick + EPA + Log_Plat + Log_Plat_De + Cortex + No_Scars,
   Reg_Data,
   method = 'neuralnet',
   trControl = train.control,
   tuneGrid = tune.grid.neuralnet,
   preProcess = c("center", "scale"),
   learningrate = 0.01,  
   threshold = 0.01,
   stepmax = (10^100),
   linear.output = TRUE
 )
```

 

Cartesian grid search of ANN topology indicates that increasing the
number of nodes in the first hidden layer decreases linear correlation
with the outcome. On general Cartesian grid search of ANN topology
indicates that increasing the number of layers and nodes results in
lower values of *r*<sup>2</sup>. Thus, the most simple ANN architecture
(one hidden layer with one node) provides the highest correlation
coefficient (r2 = 0.78). The second best topology (two hidden layers
with one node at each layer) provides a marginally lower value (0.0005
lower).

``` r
data.frame(nnet_model$results) %>% 
  ggplot(aes(layer1, layer2, fill = Rsquared)) + 
  geom_tile(alpha = 0.75) +
  geom_text(aes(label = round(Rsquared, 4)), size = 3) +
  ggsci::scale_fill_gsea(reverse = TRUE) +
  xlab("Number of nodes in layer 1") +
  ylab("Number of nodes in layer 2") +
  theme_classic() +
  coord_fixed() +
  theme(legend.position = "none",
        axis.text = element_text(color = "black", size = 8))
```

![](01-Complete-script_files/figure-markdown_github/Plot%20of%20ANN%20performance%20for%20diferent%20topologies-1.png)

 

## 04 Model evaluation

### 04.1 Model evaluation metrics

The following table presents the precision metrics for each model. On
general ANN and multiple linear regression perform similarly with
similar values of r2 (0.78), RMSE (0.21) and MAE (0.17), although ANN
performs slightly better. On the other hand Random Forest regression
performs slightly worst with a lower value of r2 (0.72) and higher
values of RMSE (0.24) and MAE (0.19).

``` r
Temp <- data.frame(rbind(
  MLR_model$results[2:4],
  nnet_model_f$results[4:6],
  RF_model$results[4:6]))

Temp <- cbind(data.frame(model = c("MLR", "ANN", "RF")), Temp)

kable(Temp)
```

| model |      RMSE |  Rsquared |       MAE |
|:------|----------:|----------:|----------:|
| MLR   | 0.2088634 | 0.7761463 | 0.1661431 |
| ANN   | 0.2086654 | 0.7766675 | 0.1656524 |
| RF    | 0.2392614 | 0.7212704 | 0.1924296 |

### 04.2 Variable importance
