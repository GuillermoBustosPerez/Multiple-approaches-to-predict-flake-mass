# Multiple approaches to predict flake mass

-   Guillermo Bustos-Pérez <sup>(1,2)</sup>  
-   Javier Baena Preysler <sup>(1)</sup>

<sup>1</sup> Departamento de Prehistoria y Arqueología, Universidad
Autónoma de Madrid, Madrid, Spain  
<sup>2</sup> Corresponding author at:
<guillermo.bustos@estudiante.uam.es> \| <guillermo.willbustos@gmail.com>

## 01 Table of contents

-   01 Installing packages  
-   02 Loading and describing the data  
-   03 Model training:
    -   03.1 Multiple Linear Regression  
    -   03.2 Random Forest Regression  
    -   03.3 Artificial Neuronal Network (ANN)  
-   04 Model evaluation
    -   04.1 Hyperparmeter grid search  
    -   04.2

``` r
# Load the data
Reg_Data <- read.csv("Data/Flake Mass v02 Eng.csv")
```

``` r
## load("Data/Reg_Data.RData")
library(neuralnet); library(tidyverse); library(caret)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v ggplot2 3.3.5     v purrr   0.3.4
    ## v tibble  3.1.5     v dplyr   1.0.7
    ## v tidyr   1.1.4     v stringr 1.4.0
    ## v readr   2.0.2     v forcats 0.5.1

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::compute() masks neuralnet::compute()
    ## x dplyr::filter()  masks stats::filter()
    ## x dplyr::lag()     masks stats::lag()

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
#### Select columns and variables #####
Reg_Data <- Reg_Data %>% 
  select(Log_Weight,
         MeanThick, Log_Max_Thick, EPA, Log_Plat, Log_Plat_De, Cortex, No_Scars)
```
