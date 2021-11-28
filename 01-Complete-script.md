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
-   03 Model training:
    -   03.1 Multiple Linear Regression  
    -   03.2 Random Forest Regression  
    -   03.3 Artificial Neuronal Network (ANN)  
-   04 Model evaluation
    -   04.1 Hyperparmeter grid search  
    -   04.2 Model evaluation  
    -   04.3 Variable importance

``` r
# Load the data
Reg_Data <- read.csv("Data/Flake Mass v02 Eng.csv")
```

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
list.of.packages <- c("tidyverse", "caret",  "ranger", "knitr")

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

 

## 02 Loading and describing the data

Sample for analysis is composed of 500 experimentally knapped flakes
using hard hammer. Flakes belong to 30 knapping sequences where a wide
variety of knapping methods were employed —hierarchical (Levallois and
Hierarchical Discoid), bifacial (Discoid), and unipolar— to generate the
experimental sample, ensuring a wide range of morphologies.

``` r
## load("Data/Reg_Data.RData")
library(neuralnet); library(tidyverse); library(caret)
```

    ## 
    ## Attaching package: 'neuralnet'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     compute

``` r
#### Select columns and variables #####
Reg_Data <- Reg_Data %>% 
  select(Log_Weight,
         MeanThick, Log_Max_Thick, EPA, Log_Plat, Log_Plat_De, Cortex, No_Scars)
```
