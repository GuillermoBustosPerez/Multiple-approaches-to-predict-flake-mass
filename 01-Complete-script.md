# Multiple approaches to predict flake mass

-   Guillermo Bustos-Pérez <sup>(1,2)</sup>  
-   Javier Baena Preysler <sup>(1)</sup>

<sup>1</sup> Departamento de Prehistoria y Arqueología, Universidad
Autónoma de Madrid, Madrid, Spain  
<sup>2</sup> Corresponding author at:
<guillermo.bustos@estudiante.uam.es> \| <guillermo.willbustos@gmail.com>

## Abstract

Predicting original flake mass is a major goal of lithic analysis.
Predicting original flake mass allows to make estimations of remaining
mass, lost mass, etc. All these measures relate to the organization of
lithic technology by past societies. The present work tests three
different models to predict log of flake mass: Multiple Linear
Regression, Random Forest regression and Artificial Neuronal Network
(ANN). Estimations of flake mass are performed using remaining features
of flakes from an experimental assemblage. This assemblage has been
obtained by the expansion of a previous dataset by the inclusion of
bigger flakes, allowing to account for the effects of sample size and
value distribution. Correlation results show a large/strong relation
between predictions and real outcome (r2 = 0.78 in the best case).
Comparison of models allows to gain insights into variable importance
for predicting flake mass. Results also show that (for the present
dataset) multiple linear regression still stands as the best method for
predicting log of flake weight. Additionally, transformation of
predicted values from the multiple linear regression and true values to
the linear scale reinforces the linear correlation above the 0.8
threshold.

**Key words:** lithic technology; experimental archaeology; flake
weight; Machine Learning; Deep Learning

## Introduction

“Curated” is a key concept for the analysis of lithic technological
organization ([Andrefsky, 2009](#ref-andrefsky_analysis_2009); [Binford,
1979](#ref-binford_organization_1979); [Nelson,
1991](#ref-nelson_study_1991); [Spry and Stern,
2016](#ref-spry_technological_2016)). Initially “curated” was defined to
encompass a series of behavioral patterns related to provisioning
strategies ([Binford, 1979](#ref-binford_organization_1979),
[1973](#ref-renfrew_interassemblage_1973)). Further authors included
tool transport, utilization in a wide range of tasks, anticipated
production, hafting and recycling (after original tool had been
discarded) into the behavioral adaptive strategies that defined a
curation. [Shott](#ref-shott_exegesis_1996)
([1996](#ref-shott_exegesis_1996), [1989](#ref-shott_tool-class_1989))
proposed an alternative interpretation of the term “curation” as the
“ratio of realized to potential utility.” This shift in the definition
of the “curation” has deep implications for lithic analysis and the
study of lithic technological organization since it transforms
“curation” into a continuous variable ([Shott,
1996](#ref-shott_exegesis_1996)). This shift of “curation” into a
continuous variable usually implies the degree of reduction or
maintenance undergone by a tool ([Shott, 2007](#ref-shott_role_2007),
[1996](#ref-shott_exegesis_1996), [1989](#ref-shott_tool-class_1989)).
Additionally, the understanding of curation as a continuum also relates
to the reduction approach ([Dibble,
1987](#ref-dibble_interpretation_1987),
[1987](#ref-dibble_interpretation_1987); [Rolland and Dibble,
1990](#ref-rolland_new_1990)) which considers processes of resharpening
as a major factor driving the presence and frequency of tool types.
Ethnographic studies also emphasize the role of retouch on resharpening
dulled edges, changes in morphology, or variations in artifact use as
morphology changes throughout reduction ([Casamiquela,
1978](#ref-casamiquela_temas_1978); [Gould,
1968](#ref-gould_living_1968); [Nuevo Delaunay et al.,
2017](#ref-nuevo_delaunay_glass_2017); [Shott and Weedman,
2007](#ref-shott_measuring_2007); [White,
1967](#ref-white_ethno-archaeology_1967)).  
Usually two approaches are employed to estimate the reduction and
curation undergone by a retouched artifact. The first branch of approach
focuses on estimations made through measurements directly made on
retouch. This has led to the proposal of several indexes which use
different measurements such as height of retouch, length of retouched
edge, or projection of original angle ([Bustos-Pérez and Baena,
2019](#ref-bustos-perez_exploring_2019); [Eren et al.,
2005](#ref-eren_defining_2005); [Hiscock and Clarkson,
2005](#ref-hiscock_experimental_2005); [Kuhn,
1990](#ref-kuhn_geometric_1990); [Morales et al.,
2015](#ref-morales_measuring_2015)). Although proposed indexes from this
broad approach usually present high correlation values, they are
conditioned by flake morphology, direction of retouch or tool type
(laterally retouched scrapers, endscrapers, bifacial products, etc.).
[Dibble](#ref-dibble_middle_1995) ([1995](#ref-dibble_middle_1995))
noted the “flat flake problem” when applying
[Kuhn](#ref-kuhn_geometric_1990) ([1990](#ref-kuhn_geometric_1990))
general index of unifacial reduction (GIUR). The “flat flake problem”
states that a flake with trapezoidal cross section (where the dorsal
face is mainly flat) will promptly reach maximum values of GIUR although
reduction continues. The effects of the “flat flake problem” don’t seem
to be so severe on the GIUR ([Hiscock and Clarkson,
2005](#ref-hiscock_experimental_2005)) but they exemplify the possible
limitations that these indexes may undergone as a result of flake
morphology. [Shott](#ref-shott_reduction_2005)
([2005](#ref-shott_reduction_2005)) extensive review of methods outlines
the strengths and limitations derived from geometry, flake morphology
and assemblage suitability faced by each of the indexes.  
The second branch of approach aims to estimate original flake mass based
on remaining features. This branch of approach has the advantage of not
being conditioned by tool type, direction of retouch, or flake
morphology. Estimating original mass and comparing it with remaining
mass can provide highly useful measures such as percentage of mass
remaining, amount of mass lost, etc. All these measures relate to the
curation concept as a continuous and the reduction approach. Initial
controlled experiments showed highly promising results on the ability to
predict flake mass from remaining features ([Dibble and Pelcin,
1995](#ref-dibble_effect_1995)). However following experiments based on
the replication of knapping methods failed to obtain such high levels of
correlation ([Davis and Shea, 1998](#ref-davis_quantifying_1998); [Shott
et al., 2000](#ref-shott_flake_2000)) Additionally in some occasions
estimated original mass was lower than mass of flake after retouch
([Davis and Shea, 1998](#ref-davis_quantifying_1998)). This posed an
important drawback since as [Dibble](#ref-dibble_comment_1998)
([1998](#ref-dibble_comment_1998)) states and [Shott et
al.](#ref-shott_flake_2000) ([2000](#ref-shott_flake_2000)) reiterates:
controlled experiments are useful only if results and variable
relationships are extendible to the archaeological record. Further
research has explored the estimation of flake mass using the combination
of several variables ([Dogandžić et al.,
2015](#ref-dogandzic_edge_2015); [Shott and Seeman,
2017](#ref-shott_use_2017)) and the determination of best variables to
perform estimations ([Bustos-Pérez and Baena,
2021](#ref-bustos-perez_predicting_2021)). [Hiscock and
Tabrett](#ref-hiscock_generalization_2010)
([2010](#ref-hiscock_generalization_2010)) state the logical and
analytical characteristics desirable for an index: inferential power;
directionality; comprehensiveness; sensitivity; versatility, blank
diversity and scale independence. Following these characteristics it can
be stated that the first branch of approach is strong in inferential
power, directionality, comprehensiveness and sensitivity. On the other
hand present estimations of flake mass are strong in inferential power,
comprehensiveness, sensitivity, versatility, blank diversity and scale
independence.

Most analysis focus on the use linear regression (usually through
platform surface as a proxy of flake mass) or the combination of several
variables in multiple linear regression. The generalization of
statistical programming software ([R. C. Team,
2019](#ref-r_core_team_r_2019); [Rs. Team,
2019](#ref-rstudio_team_rstudio_2019)) allows for the implementation of
regression models beyond the simple linear regression. The present study
uses and evaluates three common Machine Learning regression models
(Artificial Neural Networks; Multiple Linear Regression and Random
Forest) for the estimation of flake mass. Additionally each model
provides insights into variable importance.

## 01 Installing packages

The following code provides the list of packages employed in the
analysis, checks if they are missing and installs the missing ones. This
is set to meet reproducibility standards for machine learning (Heil et
al., 2021).

``` r
list.of.packages <- c("tidyverse", "lattice", "caret", "neuralnet", "ranger", "NeuralNetTools")

new.packages <- list.of.packages[!(list.of.packages %in% 
                                     installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages)
```

 

After this we can load the packages to perform model training and
analysis. Additionally in this markdown we are going to use package
knitr to show a nice output of tables. The present study makes extensive
use of tidyverse (Wickham et al., 2019) and caret (Kuhn, 2008) for the
treatment of data and training of models.

``` r
list.of.packages <- c("tidyverse", "lattice", "caret", "neuralnet", "ranger", "knitr")

lapply(list.of.packages, library, character.only = TRUE)
```

    ## [[1]]
    ##  [1] "forcats"   "stringr"   "dplyr"     "purrr"     "readr"     "tidyr"    
    ##  [7] "tibble"    "ggplot2"   "tidyverse" "stats"     "graphics"  "grDevices"
    ## [13] "utils"     "datasets"  "methods"   "base"     
    ## 
    ## [[2]]
    ##  [1] "lattice"   "forcats"   "stringr"   "dplyr"     "purrr"     "readr"    
    ##  [7] "tidyr"     "tibble"    "ggplot2"   "tidyverse" "stats"     "graphics" 
    ## [13] "grDevices" "utils"     "datasets"  "methods"   "base"     
    ## 
    ## [[3]]
    ##  [1] "caret"     "lattice"   "forcats"   "stringr"   "dplyr"     "purrr"    
    ##  [7] "readr"     "tidyr"     "tibble"    "ggplot2"   "tidyverse" "stats"    
    ## [13] "graphics"  "grDevices" "utils"     "datasets"  "methods"   "base"     
    ## 
    ## [[4]]
    ##  [1] "neuralnet" "caret"     "lattice"   "forcats"   "stringr"   "dplyr"    
    ##  [7] "purrr"     "readr"     "tidyr"     "tibble"    "ggplot2"   "tidyverse"
    ## [13] "stats"     "graphics"  "grDevices" "utils"     "datasets"  "methods"  
    ## [19] "base"     
    ## 
    ## [[5]]
    ##  [1] "ranger"    "neuralnet" "caret"     "lattice"   "forcats"   "stringr"  
    ##  [7] "dplyr"     "purrr"     "readr"     "tidyr"     "tibble"    "ggplot2"  
    ## [13] "tidyverse" "stats"     "graphics"  "grDevices" "utils"     "datasets" 
    ## [19] "methods"   "base"     
    ## 
    ## [[6]]
    ##  [1] "knitr"     "ranger"    "neuralnet" "caret"     "lattice"   "forcats"  
    ##  [7] "stringr"   "dplyr"     "purrr"     "readr"     "tidyr"     "tibble"   
    ## [13] "ggplot2"   "tidyverse" "stats"     "graphics"  "grDevices" "utils"    
    ## [19] "datasets"  "methods"   "base"

 

## 02 Loading and describing the data

Sample for analysis is composed of 500 experimentally knapped flakes
using a variety of hard hammerd (quartzite, quartz, sandstone). Flakes
belong to 30 knapping sequences where a wide variety of knapping methods
were employed —hierarchical (Levallois and Hierarchical Discoid),
bifacial (Discoid), and unipolar— to generate the experimental sample,
ensuring a wide range of morphologies. This is an expansion of a
previous dataset employed for similar purposes (Bustos-Pérez and Baena,
2021) which allows to expand the range of dimensions and mass of the
assemblage.

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

 

A fast way to explore lithic assemblage composition is through a
Bagolini scatter plot (Bagolini, 1968). Comparison of the experimental
dataset with the one from the previous study (Bustos-Pérez and Baena,
2021) shows an increase on the size and average mass of experimentally
knapped flakes. While in the previous study 50% of the flakes had mass
values between 4.15g and 14.02g (Bustos-Pérez and Baena, 2021), in the
present study 50% of the flakes weight between 5.87g and 26.96g. This
indicates that the expansion of the dataset has been done by the
inclusion of heavier and bigger flakes.

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
 

Collinearity between predictors has previously been reported for
platform surface and platform depth, and mean thickness and log10 of
maximum thickness (Bustos-Pérez and Baena, 2021). For the present
dataset there is an important collinearity between log10 of maximum
thickness and mean thickness (*r*<sup>2</sup> = 0.879); and an expected
moderate/strong collinearity between platform depth and platform surface
(*r*<sup>2</sup> = 0.614). Been aware of these collinearities is
important since collinearity affects variable importance (hard to
separate the individual effect of a variable on the response), it
reduces the accuracy of the estimates on a Multiple Linear Regression,
and it can result in counterintuitive estimates (James et al., 2013).
Despite the challenges collinearity poses it is important to consider
that collinearity does not affect predictions and the inferential power
of the model (Alin, 2010; Paul, 2006). The maine focus of the present is
the predictive accuracy of the models and not in the relations between
predictors and dependent variable.

``` r
# Collinearity between measures of thickness
R2(Reg_Data$MeanThick, Reg_Data$Log_Max_Thick)
```

    ## [1] 0.8791263

``` r
# Collinearity between measures of platform
R2(Reg_Data$Platfom_Depth, Reg_Data$Surface.Plat)
```

    ## [1] 0.6140689

## 03 Model training and hyperparameter tunning

### 03.1 Multiple Linear regression

Multiple linear regression extends the simple linear regression to
accommodate multiple predictors:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> +  ·  ·  · *β*<sub>*p*</sub>*X*<sub>*p*</sub> + *ϵ*
 

Althought Multiple Linear Regression is less prone to overfit the data
it is still a good practice to performe multiple k-fold cross
validation. As a standard the present research uses a 10 fold cross
validation with 50 cycles.

``` r
# Set Train control
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
                       trControl = train.control,
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

RF_weight <- train(frmla, 
                   Reg_Data,
                   method = "ranger",
                   trControl = train.control,
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
coefficient (*r*<sup>2</sup> = 0.78). The second best topology (two
hidden layers with one node at each layer) provides a marginally lower
value (0.0005 lower).

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

 

## 04 Results

### 04.1 Model evaluation metrics

The following table presents the precision metrics for each model. On
general ANN and multiple linear regression perform similarly with
similar values of *r*<sup>2</sup> (0.78), RMSE (0.21) and MAE (0.17),
although ANN performs slightly better. On the other hand Random Forest
regression performs slightly worst with a lower value of *r*<sup>2</sup>
(0.72) and higher values of RMSE (0.24) and MAE (0.19).

``` r
Temp <- data.frame(rbind(
  round(MLR_model$results[2:4],2),
  round(nnet_model_f$results[4:6],2),
  round(RF_model$results[4:6],2)))

Temp <- cbind(data.frame(model = c("MLR", "ANN", "RF")), Temp)

kable(Temp)
```

| model | RMSE | Rsquared |  MAE |
|:------|-----:|---------:|-----:|
| MLR   | 0.21 |     0.78 | 0.17 |
| ANN   | 0.21 |     0.78 | 0.17 |
| RF    | 0.24 |     0.72 | 0.19 |

 

Visualization of regression plots for each model provides additional
information of the performance of each model. The poor performance of
Random Forest (lowest value of *r*<sup>2</sup>) is reflected in a
limited range of prediction. The prediction range of the Random Forest
is limited between a minimum value of 0.55 and a maximum value of 1.76
for log10 of flake mass. As a result of this, data is not evenly
distributed among the regression line. In the lowest values of
prediction most points fall below the regression line while most data
points falling above for the highest values of the regression line. ANN
and multiple linear regression plots present similar patterns of
distribution with data evenly distributed among the regression line.
Flakes with a log10 value of flake mass above 2 are lightly more evenly
distributed for the multiple linear regression than for the ANN.

``` r
#### Linear model ####
MLR_results <- as.data.frame(MLR_model$pred) %>% 
  group_by(rowIndex) %>% 
  summarise(Pred = mean(pred),
            Obs = mean(obs)) %>% 
  mutate(Residual = Obs - Pred)

#### ANN model ####
nnet_results <- as.data.frame(nnet_model_f$pred) %>% 
  group_by(rowIndex) %>% 
  summarise(Pred = mean(pred),
            Obs = mean(obs)) %>% 
  mutate(Residual = Obs - Pred)

#### RF Model ####
RF_results <- as.data.frame(RF_model$pred) %>% 
  group_by(rowIndex) %>% 
  summarise(Pred = mean(pred),
            Obs = mean(obs)) %>% 
  mutate(Residual = Obs - Pred)

#### Put models together #####
Temp <- rbind(MLR_results, nnet_results, RF_results)
Temp$Model <- "Multiple linear regression"
Temp$Model[501:1000] <- "ANN"
Temp$Model[1001:1500] <- "Random Forest"

#### Correlation plot ####
Temp %>% 
  ggplot(aes(Pred, Obs)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_line(aes(y = Pred), size = 1, col = "blue") +
  
  scale_x_continuous(breaks = seq(0, 2.55, 0.25), lim = c(-0.1, 2.35)) +
  scale_y_continuous(breaks = seq(0, 2.55, 0.25), lim = c(-0.1, 2.35)) +
  
  xlab("Predicted") +
  ylab("Observed") +

  facet_wrap(~ Model, ncol = 3) + 
  coord_fixed() +
  theme_light() +
  theme(strip.text = element_text(color = "black", face = "bold", size = 9),
        strip.background = element_rect(fill = "white", colour = "black", size = 1),
        axis.text = element_text(size = 7.5, color = "black"))
```

![](01-Complete-script_files/figure-markdown_github/Regression%20plots%20of%20all%20models-1.png)
 

### 04.2 Residuals analysis and distribution

Visual analysis of the scatter plot for observed and residual values
allows to observe model performance for different ranges of log10 of
flake mass values. Residuals of the Random Forest present a systematic
bias for the upper and lowest values of observed weight. In the case of
flakes with a log10 value of 0.50 there is a systematic overestimation
of the size. In the case of flakes with a log10 value of 1.75 there is a
systematic underestimation of values. ANN and multiple linear regression
present very similar plots for observed values and residuals. In both
cases residual values indicate a systematic overestimation of a log10
flake mass when the actual value is below 0.25.  
Between values of 0.25 and 2 both models present a very similar
performance with residual values falling evenly among the 0 value. ANN
seems to present a slightly systematic underestimation of flakes with a
log10 of flake mass above a value of 2. Multiple linear regression does
seem to perform better for flakes with a log10 flake mass value above 2
with residual values falling evenly or very close to the 0 value line.

``` r
## Plots of observed values and residuals
Temp %>% 
  ggplot(aes(Obs, Residual)) +
  geom_point(alpha = 0.5, size = 1.5) +
  
  xlab("Observed") +
  ylab("Residual") +
  
  scale_x_continuous(breaks = seq(0, 2.55, 0.25), lim = c(min(Temp$Obs), max(Temp$Obs))) +
  scale_y_continuous(breaks = seq(-0.75, 0.75, 0.25), lim = c(-0.85, 0.85)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  
  facet_wrap(~ Model, ncol = 3) + 
  coord_fixed() +
  theme_light() +
  theme(strip.text = element_text(color = "black", face = "bold", size = 9),
        strip.background = element_rect(fill = "white", colour = "black", size = 1),
        axis.text = element_text(size = 7.5, color = "black"))
```

![](01-Complete-script_files/figure-markdown_github/Plots%20of%20observed%20values%20and%20residuals-1.png)
 

Correlation between observed values and residuals allows to evaluate if
residuals increase along with increasing values of log10 of weight. ANN
and multiple linear regression models present the same value of
*r*<sup>2</sup> for correlation of observed values and residuals
(*r*<sup>2</sup> = 0.22; p \< 0.01) while Random Forest presents a
higher value of correlation (*r*<sup>2</sup> = 0.5; p \< 0.01).

``` r
# Residuals and multiple linear regression
summary(lm(Residual ~ Obs, MLR_results))
```

    ## 
    ## Call:
    ## lm(formula = Residual ~ Obs, data = MLR_results)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.54486 -0.11287  0.00723  0.12352  0.53357 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -0.24837    0.02260  -10.99   <2e-16 ***
    ## Obs          0.22306    0.01889   11.81   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1857 on 498 degrees of freedom
    ## Multiple R-squared:  0.2188, Adjusted R-squared:  0.2172 
    ## F-statistic: 139.4 on 1 and 498 DF,  p-value: < 2.2e-16

``` r
# Residuals and ANN
summary(lm(Residual ~ Obs, nnet_results))
```

    ## 
    ## Call:
    ## lm(formula = Residual ~ Obs, data = nnet_results)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.55719 -0.10717 -0.00036  0.13051  0.51214 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -0.24798    0.02258  -10.98   <2e-16 ***
    ## Obs          0.22286    0.01887   11.81   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1855 on 498 degrees of freedom
    ## Multiple R-squared:  0.2188, Adjusted R-squared:  0.2172 
    ## F-statistic: 139.5 on 1 and 498 DF,  p-value: < 2.2e-16

``` r
# Residuals and RF model
summary(lm(Residual ~ Obs, RF_results))
```

    ## 
    ## Call:
    ## lm(formula = Residual ~ Obs, data = RF_results)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.49052 -0.11216  0.00151  0.10916  0.47628 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -0.42164    0.02052  -20.55   <2e-16 ***
    ## Obs          0.38224    0.01715   22.29   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1686 on 498 degrees of freedom
    ## Multiple R-squared:  0.4994, Adjusted R-squared:  0.4984 
    ## F-statistic: 496.8 on 1 and 498 DF,  p-value: < 2.2e-16

 

Descriptive statistics of residuals and density plots allow to evaluate
dispersion range of residuals. All models present average and median
residual values close to 0 with density curves peaking near this value
which is indicative of a good model performance. 50% of residual values
from ANN model fall between values of -0.133 and 0.143 making for a
distance of 0.276. 50% of residual values from multiple linear
regression model fall between values of -0.137 and 0.134 making for a
distance of 0.271. 50% of residual values from multiple Random Forest
model fall between values of -0.138 and 0.177 making for a distance of
0.315. This indicates that multiple linear regression model concentrates
50% of residuals values in a slightly shorter range. This range is 0.005
shorter than the one from ANN model. The Random forest presents the
highest dispersion range for 50% of residual values.

``` r
# Density plot of residuals
Temp %>% ggplot(aes(Residual, color = Model)) +
  geom_density(size = 1) +
  ggsci::scale_color_aaas() +
  scale_x_continuous(breaks = seq(-1, 1, 0.25), lim = c(-1,1)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_hline(yintercept = 0) +
  ylab("Density") +
  theme_light() +
  theme(legend.position = "bottom",
        legend.title = element_text(face = "bold"),
        axis.text = element_text(color = "black", size = 9),
        axis.title = element_text(color = "black", size = 10))
```

![](01-Complete-script_files/figure-markdown_github/Density%20plot%20of%20residuals-1.png)

90% of residual values from ANN model fall between values of -0.379 and
0.33 making for a distance of 0.709. 90% of residual values from
multiple linear regression model fall between values of -0.371 and 0.333
making for a distance of 0.704. 90% of residual values from Random
Forest model fall between values of -0.412 and 0.355 making for a
distance of 0.767. Again, multiple linear regression concentrates 90% of
residuals in the shortest range. ANN presents a slightly wider range
(difference of 0.005) and Random Forest presents the widest range of the
three models.

``` r
kable(
  Temp %>% group_by(Model) %>% 
  summarise(
    Min = min(Residual),
    `5 Percentil` = quantile(Residual, 0.05),
    `1Quantile` = quantile(Residual, 0.25),
    Mean = mean(Residual),
    Median = quantile(Residual, 0.5),
    `3Quantile` = quantile(Residual, 0.75),
    `95 Percentil` = quantile(Residual, 0.95),
    Max = max(Residual)
  ))
```

| Model                      |        Min | 5 Percentil |  1Quantile |       Mean |    Median | 3Quantile | 95 Percentil |       Max |
|:---------------------------|-----------:|------------:|-----------:|-----------:|----------:|----------:|-------------:|----------:|
| ANN                        | -0.6952028 |  -0.3790468 | -0.1331217 |  0.0000595 | 0.0240733 | 0.1428723 |    0.3295823 | 0.4993439 |
| Multiple linear regression | -0.7046012 |  -0.3711780 | -0.1368752 | -0.0001095 | 0.0204098 | 0.1341937 |    0.3332873 | 0.4823185 |
| Random Forest              | -0.8456258 |  -0.4117285 | -0.1383524 |  0.0037893 | 0.0133008 | 0.1773731 |    0.3550391 | 0.6154902 |

 

``` r
# Order of flakes is kept the same for all models
Terminations <- Reg_Data %>% select(Termination_type)
Terminations <- rbind(Terminations, Terminations, Terminations)
Terminations <- cbind(Temp, Terminations)

# Mutate to new categories: feather termination and other types of terminations
Terminations <- Terminations %>% mutate(
  New_Term =
  case_when(
    Termination_type == "Feather" ~ "Feather",
    Termination_type != "Feather" ~ "Other"))
```

``` r
Terminations %>% 
  ggplot(aes(New_Term, Residual, fill = New_Term)) +
  facet_wrap(~ Model, ncol = 3) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(alpha = 0.8, outlier.size = 0) +
  geom_jitter(width = 0.15, alpha = 1, size = 0.9, shape = 23, aes(fill = New_Term)) +
  ggsci::scale_fill_aaas() +
  xlab(NULL) +
  theme_light() +
  theme(
    legend.position = "none",
    strip.text = element_text(color = "black", face = "bold", size = 9),
    strip.background = element_rect(fill = "white", colour = "black", size = 1),
    axis.text = element_text(color = "black", size = 8.5),
    axis.title = element_text(color = "black", size = 9))
```

![](01-Complete-script_files/figure-markdown_github/residuals%20according%20to%20termination-1.png)

 

Exploratory data analysis of residuals according to termination type
through box and violin plots shows possible differences in the
distribution for the three models. Comparison of residuals means
according to termination type and for each model through t-test shows
significant differences for the ANN model (t = -2.5; p = 0.02), the
multiple linear regression (t = -2.52; p = 0.01), but not for the random
forest regression (t = -1.82, p = 0.07). In all models residuals mean of
flakes with feather terminations fall near the cero value (-0.007 in the
case of ANN; -0.008 in the case of multiple linear regression and -0.002
in the case of Random Forest). Flakes with other termination than
feather tend to have a slightly higher mean of residuals values (0.07 in
the case of ANN; 0.07 in the case of multiple linear regression; 0.06 in
the case of random Forest).

``` r
# t-test residuals and terminations for ANN
t.test(Residual ~ New_Term, data = Terminations[Terminations$Model == "ANN",])
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  Residual by New_Term
    ## t = -2.4964, df = 63.439, p-value = 0.01516
    ## alternative hypothesis: true difference in means between group Feather and group Other is not equal to 0
    ## 95 percent confidence interval:
    ##  -0.13238703 -0.01467883
    ## sample estimates:
    ## mean in group Feather   mean in group Other 
    ##           -0.00744086            0.06609207

``` r
# t-test residuals and terminations for Multiple linear regression
t.test(Residual ~ New_Term, data = Terminations[Terminations$Model == "Multiple linear regression",])
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  Residual by New_Term
    ## t = -2.5239, df = 62.665, p-value = 0.01416
    ## alternative hypothesis: true difference in means between group Feather and group Other is not equal to 0
    ## 95 percent confidence interval:
    ##  -0.13631906 -0.01583418
    ## sample estimates:
    ## mean in group Feather   mean in group Other 
    ##          -0.007869329           0.068207291

``` r
# t-test residuals and terminations for Random Forest
t.test(Residual ~ New_Term, data = Terminations[Terminations$Model == "Random Forest",])
```

    ## 
    ##  Welch Two Sample t-test
    ## 
    ## data:  Residual by New_Term
    ## t = -1.8155, df = 65.4, p-value = 0.07403
    ## alternative hypothesis: true difference in means between group Feather and group Other is not equal to 0
    ## 95 percent confidence interval:
    ##  -0.121451961  0.005779731
    ## sample estimates:
    ## mean in group Feather   mean in group Other 
    ##          -0.002109984           0.055726131

 

### 04.2 Variable importance

The following presents variable relative importance scaled from 0 to 100
for each model. ANN considers of key importance variables of thickness
(mean thickness of the flake and log10 of flake thickness) along with
log10 of platform size. Relative amount of cortex is also considered to
have relative importance. Multiple linear regression considers relative
amount of cortex, number of scars and log10 of platform size as the most
important variables. Log10 of maximum thickness is also considered as an
important variable. Random Forest only considers mean thickness and
log10 of maximum thickness as important variables.

Only ANN model provides an importance value above 0 for EPA, although it
is considered as the least important variable for that model. On general
log10 of platform size is considered as an important variable being the
third most important variable for multiple linear regression and Random
Forest; and the second most important variable for ANN. Log10 of
platform depth is usually considered a minor important variable by all
models with scores of importance below the 50 value threshold.

``` r
#### Variable importance ####
# Make Data frame of importance of MLR
mr_imp <- varImp(MLR_model, scale = TRUE)
mr_imp <- as.data.frame(mr_imp$importance)
mr_imp$Variable = rownames(mr_imp)
mr_imp$Model = "Multiple Linear Regression"

# Make Data frame of importance of ANN
temp <- NeuralNetTools::garson(nnet_model_f)

ANN_imp <- data.frame(temp$data) %>% mutate(
  Overall = (rel_imp*100)/0.24145208) %>% 
  select(-c(rel_imp)) %>% 
  rename(Variable = x_names) %>% 
  mutate(Model = "ANN") %>% 
  select(Overall, Variable, Model)

# Make Data frame of RF importance
RF_imp <- varImp(RF_model, scale = TRUE)
RF_imp <- as.data.frame(RF_imp$importance)
RF_imp$Variable = rownames(RF_imp)
RF_imp$Model = "Random Forest"

#### Variable importance according to model 
Var_Imp <- rbind(mr_imp, ANN_imp, RF_imp)
rm(mr_imp, ANN_imp, RF_imp)

# reset variable labels
Var_Imp$Variable <- factor(Var_Imp$Variable,
                           levels = c("MeanThick", "Log_Max_Thick",
                                      "Log_Plat", "Log_Plat_De",
                                      "EPA", "Cortex", "No_Scars"),
                           labels = c("Mean\nThickness", "Log of\nMax. Thick.",
                                      "Log of Pla.", "Log of\nPlat. Depth",
                                    "EPA", "Cortex",
                                    "No Scars"))

# and plot
Var_Imp %>% ggplot(aes(Variable, Overall, fill = Overall)) +
  geom_col() +
  geom_text(data = Var_Imp[Var_Imp$Overall > 0,],
            aes(label = round(Overall, 2)), vjust= "top", size = 2.5) +
  ggsci::scale_fill_gsea(reverse = TRUE) +
  theme_classic() +
  facet_wrap(~ Model, ncol = 1) +
  ylab("Relative importance") +
  xlab(NULL) +
  theme(legend.position = "none",
        axis.text = element_text(color = "black", size = 7),
        strip.text = element_text(color = "black", face = "bold", size = 8),
        strip.background = element_rect(fill = "white", colour = "black", size=1),
        axis.title.y = element_text(color = "black", size = 8))
```

![](01-Complete-script_files/figure-markdown_github/extract%20variable%20importance%20and%20plot-1.png)

 

### 04.4 Linear transformation of predictions

The following table presents the performance metrics of each model after
transforming true and predicted values back to the linear scale. ANN and
multiple linear regression reinforce their correlation while Random
Forest decreases it’s *r*<sup>2</sup> value. Multiple linear regression
provides the highest *r*<sup>2</sup> value (*r*<sup>2</sup> = 0.813)
followed by ANN (*r*<sup>2</sup> = 0.801), indicating that multiple
linear regression generalizes better to the linear scale. All models
present lower RMSE values than the standard deviation value of weight of
the experimental assemblage (24.83) which is indicative of a good
general performance.

``` r
# Transform into linear scale
Temp <- Temp %>% 
  mutate(Observed = 10^Obs,
         Predicted = 10^Pred,
         Line_Res = Observed - Predicted)
```

``` r
ANN <- Temp %>% filter(Model == "ANN")
MLR <- Temp %>% filter(Model == "Multiple linear regression")
RF <- Temp %>% filter(Model == "Random Forest")

# Performance metrics
kable(data.frame(
  "Metric" = 
    c("r2", "RMSE", "MAE"),
  
  "ANN" = 
    c(round(R2(ANN$Observed, ANN$Predicted),3),
  round(RMSE(ANN$Observed, ANN$Predicted),3),
  round(MAE(ANN$Observed, ANN$Predicted), 3)),
  
  "Mult. Linear Reg." = 
    c(round(R2(MLR$Observed, MLR$Predicted),3),
    round(RMSE(MLR$Observed, MLR$Predicted),3),
    round(MAE(MLR$Observed, MLR$Predicted), 3)),
  
  "Random Forest" = 
    c(round(R2(RF$Observed, RF$Predicted),3),
    round(RMSE(RF$Observed, RF$Predicted),3),
    round(MAE(RF$Observed, RF$Predicted), 3))
))
```

| Metric |    ANN | Mult..Linear.Reg. | Random.Forest |
|:-------|-------:|------------------:|--------------:|
| r2     |  0.801 |             0.814 |         0.660 |
| RMSE   | 11.344 |            10.853 |        16.996 |
| MAE    |  6.942 |             6.793 |         8.700 |

 

Visualization of regression plots also supports the better
generalization of multiple linear regression to the linear scale. Random
Forest limits its maximum prediction to 57.2 g resulting in a poor
generalization to the linear scale. Due to this, residuals from the
Random Forest indicate important underestimations of flake weight with
an average underestimation of 4.6 g. 50% of the residuals of the Random
Forest range between overestimations of 2.64 g and underestimations of
7.06 g. 90% of the residuals from the random forest range between
overestimations of 10.35 g and underestimations of 29.74 g.

``` r
## Regression plot on linear scale
Temp %>% ggplot(aes(Predicted, Observed)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_line(aes(y = Predicted), size = 1, col = "blue") +
  
  scale_x_continuous(breaks = seq(0, 205, 20), lim = c(0, 205)) +
  scale_y_continuous(breaks = seq(0, 205, 20), lim = c(0, 205)) +
  
  facet_wrap(~ Model) +
  coord_fixed() +
  theme_light() +
  theme(strip.text = element_text(color = "black", face = "bold", size = 9),
        strip.background = element_rect(fill = "white", colour = "black", size = 1),
        axis.text = element_text(size = 7.5, color = "black"))
```

![](01-Complete-script_files/figure-markdown_github/unnamed-chunk-10-1.png)
 

Visual representation of residuals of the Random Forest through density
plot shows that despite peaking on the 0 value it presents a long tale
of positive residuals as a result of underestimations of predictions.

ANN generalizes better to the linear scale with a higher range of
predictions which reach a maximum value of 123 g. Density plot of
residuals from the ANN present a concentrated peak on the 0 value with a
mean value of 1.82 g. Despite this ANN residuals still present a
slightly long tale of positive values for residuals as a result of some
underestimations. 50% of residuals from ANN range between
overestimations of 2.52 g and underestimations of 5.55 g. 90% of the
residuals from ANN range between overestimations of 13.18 g and
underestimations of 18.79 g.

``` r
#  Density plot of residuals in the linear scale
Temp %>% ggplot(aes(Line_Res, color = Model)) +
  geom_density(size = 0.75) +
  ggsci::scale_color_aaas() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_hline(yintercept = 0) +
  ylab("Density") +
  xlab("Residuals (g)") +
  theme_light() +
  theme(legend.position = "bottom",
        legend.title = element_text(face = "bold",size = 9),
        axis.text = element_text(color = "black", size = 9),
        axis.title = element_text(color = "black", size = 10),
        legend.text = element_text(size = 9))
```

![](01-Complete-script_files/figure-markdown_github/Density%20plot%20of%20residuals%20in%20the%20linear%20scale-1.png)

As previously mentioned Multiple Linear Regression generalizes better to
the linear scale with a maximum predicted value of 170 g. Residuals
present an average 1.4 g value, with the density plot peaking near the 0
value and similar tales to the positive and negative values (Figure 12).
50% of residuals from Multiple Linear Regression range between
overestimations of 2.42 g and underestimations of 5.73 g. 90% of
residuals from Multiple Linear Regression range between overestimations
of 13.18 g and underestimations of 18.79 g. Thus, Multiple Linear
regression presents the concentration of 90% of residuals in the
shortest range.

``` r
# Descriptive statistics of residuals in the linear scale

kable(Temp %>% group_by(Model) %>% 
  summarise(
    Min = min(Line_Res),
    `5 Percentil` = quantile(Line_Res, 0.05),
    `1Quantile` = quantile(Line_Res, 0.25),
    Mean = mean(Line_Res),
    Median = quantile(Line_Res, 0.5),
    `3Quantile` = quantile(Line_Res, 0.75),
    `95 Percentil` = quantile(Line_Res, 0.95),
    Max = max(Line_Res)
  ))
```

| Model                      |       Min | 5 Percentil | 1Quantile |     Mean |    Median | 3Quantile | 95 Percentil |       Max |
|:---------------------------|----------:|------------:|----------:|---------:|----------:|----------:|-------------:|----------:|
| ANN                        | -47.08316 |   -13.79028 | -2.516174 | 1.816030 | 0.4762883 |  5.552758 |     19.80933 |  84.98876 |
| Multiple linear regression | -60.22261 |   -13.18240 | -2.422225 | 1.400462 | 0.3307878 |  5.724842 |     18.79086 |  55.58517 |
| Random Forest              | -20.06615 |   -10.35089 | -2.641137 | 4.611775 | 0.3338027 |  7.059517 |     29.74203 | 152.07560 |

## 05 References

Alin, A., 2010. Multicollinearity. Wiley Interdisciplinary Reviews:
Computational Statistics 2, 370–374.

Bagolini, B., 1968. Ricerche sulle dimensioni dei manufatti litici
preistorici non ritoccati. Annali dell’Università di Ferrara : nuova
serie, Sezione XV. Paleontologia Umana e Paletnologia 1, 195–219.

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32.
<https://doi.org/10.1023/A:1010933404324>

Bustos-Pérez, G., Baena, J., 2021. Predicting Flake Mass: A View from
Machine Learning. Lithic Technology 46, 130–142.
<https://doi.org/10.1080/01977261.2021.1881267>

Günther, F., Fritsch, S., 2010. Neuralnet: training of neural networks.
The R Journal 2, 30–38.

Heil, B.J., Hoffman, M.M., Markowetz, F., Lee, S.-I., Greene, C.S.,
Hicks, S.C., 2021. Reproducibility standards for machine learning in the
life sciences. Nature Methods 18, 1132–1135.

James, G., Witten, D., Hastie, T., Tibshirani, R., 2013. An Introduction
to Statistical Learning with Applications in R, Second Edition.
ed. Springer.

Kuhn, M., 2008. Building Predictive Models in R using the caret Package.
Journal of Statistical Software 28.
<https://doi.org/10.18637/jss.v028.i05>

Paul, R.K., 2006. Multicollinearity: Causes, effects and remedies.
IASRI, New Delhi 1, 58–65.

Rumelhart, D.E., Hinton, G.E., Williams, R.J., 1986. Learning
representations by back-propagating errors. Nature 323, 533–536.

Wickham, H., Averick, M., Bryan, J., Chang, W., McGowan, L., François,
R., Grolemund, G., Hayes, A., Henry, L., Hester, J., Kuhn, M., Pedersen,
T., Miller, E., Bache, S., Müller, K., Ooms, J., Robinson, D., Seidel,
D., Spinu, V., Takahashi, K., Vaughan, D., Wilke, C., Woo, K., Yutani,
H., 2019. Welcome to the Tidyverse. JOSS 4, 1686.
<https://doi.org/10.21105/joss.01686>

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-andrefsky_analysis_2009" class="csl-entry">

Andrefsky, W., 2009. The analysis of stone tool procurement, production,
and maintenance. Journal of Archaeological Research 17, 65–103.
<https://doi.org/10.1007/s10814-008-9026-2>

</div>

<div id="ref-binford_organization_1979" class="csl-entry">

Binford, L.R., 1979. Organization and formation processes: Looking at
curated technologies. Journal of Anthropological Research 35, 255–273.

</div>

<div id="ref-renfrew_interassemblage_1973" class="csl-entry">

Binford, L.R., 1973. Interassemblage variability - the mousterian and
the ’functional ’ argument, in: Renfrew, C. (Ed.), The Explanation of
Culture Change. Models in Prehistory. Duckworth, Gloucester, pp.
227–254.

</div>

<div id="ref-bustos-perez_predicting_2021" class="csl-entry">

Bustos-Pérez, G., Baena, J., 2021. Predicting flake mass: A view from
machine learning. Lithic Technology 46, 130–142.
<https://doi.org/10.1080/01977261.2021.1881267>

</div>

<div id="ref-bustos-perez_exploring_2019" class="csl-entry">

Bustos-Pérez, G., Baena, J., 2019. Exploring volume lost in retouched
artifacts using height of retouch and length of retouched edge. Journal
of Archaeological Science: Reports 27, 101922.
<https://doi.org/10.1016/j.jasrep.2019.101922>

</div>

<div id="ref-casamiquela_temas_1978" class="csl-entry">

Casamiquela, R.M., 1978. Temas patagónicos de interes arqueológico. La
talla del vidrio. Relaciones de la Sociedad Argentina de Antropología
12, 213–223.

</div>

<div id="ref-davis_quantifying_1998" class="csl-entry">

Davis, Z.J., Shea, J.J., 1998. Quantifying lithic curation: An
experimental test of dibble and pelcin’s original flake-tool mass
predictor. Journal of Archaeological Science 25, 603–610.

</div>

<div id="ref-dibble_comment_1998" class="csl-entry">

Dibble, H.L., 1998. Comment on “quantifying lithic curation: An
experimental test of dibble and pelcin’s original flake-tool mass
predictor,” by zachary j. Davis and john j. shea. Journal of
Archaeological Science 25, 611–613.
<https://doi.org/10.1006/jasc.1997.0254>

</div>

<div id="ref-dibble_middle_1995" class="csl-entry">

Dibble, H.L., 1995. Middle paleolithic scraper reduction: Background,
clarification, and review of the evidence to date. Journal of
Archaeological Method and Theory 2, 300–368.

</div>

<div id="ref-dibble_interpretation_1987" class="csl-entry">

Dibble, H.L., 1987. The interpretation of middle paleolithic scraper
morphology. American Antiquity 52, 109–117.

</div>

<div id="ref-dibble_effect_1995" class="csl-entry">

Dibble, H.L., Pelcin, A., 1995. The effect of hammer mass and velocity
on flake mass. Journal of Archaeological Science 22, 429–439.
<https://doi.org/10.1006/jasc.1995.0042>

</div>

<div id="ref-dogandzic_edge_2015" class="csl-entry">

Dogandžić, T., Braun, D.R., McPherron, S.P., 2015. Edge length and
surface area of a blank: Experimental assessment of measures, size
predictions and utility. PLoS ONE 10, e0133984.
<https://doi.org/10.1371/journal.pone.0133984>

</div>

<div id="ref-eren_defining_2005" class="csl-entry">

Eren, M.I., Domínguez-Rodrigo, M., Kuhn, S.L., Adler, D.S., Le, I.,
Bar-Yosef, O., 2005. Defining and measuring reduction in unifacial stone
tools. Journal of Archaeological Science 32, 1190–1201.
<https://doi.org/10.1016/j.jas.2005.03.003>

</div>

<div id="ref-gould_living_1968" class="csl-entry">

Gould, R.A., 1968. Living archaeology: The ngatatjaraof western
australia. Southwestern Journal of Anthropology 24, 101–122.

</div>

<div id="ref-hiscock_experimental_2005" class="csl-entry">

Hiscock, P., Clarkson, C., 2005. Experimental evaluation of kuhn’s
geometric index of reduction and the flat-flake problem. Journal of
Archaeological Science 32, 1015–1022.
<https://doi.org/10.1016/j.jas.2005.02.002>

</div>

<div id="ref-hiscock_generalization_2010" class="csl-entry">

Hiscock, P., Tabrett, A., 2010. Generalization, inference and the
quantification of lithic reduction. World Archaeology 42, 545–561.
<https://doi.org/10.1080/00438243.2010.517669>

</div>

<div id="ref-kuhn_geometric_1990" class="csl-entry">

Kuhn, S.L., 1990. A geometric index of reduction for unifacial stone
tools. Journal of Archaeological Science 17, 583–593.

</div>

<div id="ref-morales_measuring_2015" class="csl-entry">

Morales, J.I., Lorenzo, C., Vergès, J.M., 2015. Measuring retouch
intensity in lithic tools: A new proposal using 3D scan data. Journal of
Archaeological Method and Theory 22, 543–558.
<https://doi.org/10.1007/s10816-013-9189-0>

</div>

<div id="ref-nelson_study_1991" class="csl-entry">

Nelson, M.C., 1991. The study of technological organization.
Archaeological Method and Theory 57–100.

</div>

<div id="ref-nuevo_delaunay_glass_2017" class="csl-entry">

Nuevo Delaunay, A., Belardi, J.B., Carballo Marina, F., Saletta, M.J.,
De Angelis, H., 2017. Glass and stoneware knapped tools among
hunter-gatherers in southern patagonia and tierra del fuego. Antiquity
91, 1330–1343. <https://doi.org/10.15184/aqy.2017.125>

</div>

<div id="ref-rolland_new_1990" class="csl-entry">

Rolland, N., Dibble, H.L., 1990. A new synthesis of middle paleolithic
variability. American Antiquity 55, 480–499.

</div>

<div id="ref-shott_role_2007" class="csl-entry">

Shott, M.J., 2007. The role of reduction analysis in lithic studies.
Lithic Technology 32, 131–141.

</div>

<div id="ref-shott_reduction_2005" class="csl-entry">

Shott, M.J., 2005. The reduction thesis and its discontents: Overview of
the volume, in: Clarkson, C., Lamb, L. (Eds.), Lithics “down Under”:
Australian Perspectives on Lithic Reduction, Use and Classification, BAR
International Series. BAR Publishing, pp. 109–125.

</div>

<div id="ref-shott_exegesis_1996" class="csl-entry">

Shott, M.J., 1996. An exegesis of the curation concept. Journal of
Anthropological Research 52, 259–280.

</div>

<div id="ref-shott_tool-class_1989" class="csl-entry">

Shott, M.J., 1989. On tool-class use lives and the formation of
archaeological assemblages. American Antiquity 54, 9–30.
<https://doi.org/10.2307/281329>

</div>

<div id="ref-shott_flake_2000" class="csl-entry">

Shott, M.J., Bradbury, A.P., Carr, P.J., Odell, G.H., 2000. Flake size
from platform attributes: Predictive and empirical approaches. Journal
of Archaeological Science 27, 877–894.
<https://doi.org/10.1006/jasc.1999.0499>

</div>

<div id="ref-shott_use_2017" class="csl-entry">

Shott, M.J., Seeman, M.F., 2017. Use and multifactorial reconciliation
of uniface reduction measures: A pilot study at the nobles pond
paleoindian site. American Antiquity 82, 723–741.
<https://doi.org/10.1017/aaq.2017.40>

</div>

<div id="ref-shott_measuring_2007" class="csl-entry">

Shott, M.J., Weedman, K.J., 2007. Measuring reduction in stone tools: An
ethnoarchaeological study of gamo hidescrapers from ethiopia. Journal of
Archaeological Science 34, 1016–1035.
<https://doi.org/10.1016/j.jas.2006.09.009>

</div>

<div id="ref-spry_technological_2016" class="csl-entry">

Spry, C., Stern, N., 2016. Technological organization, in: Jackson, J.L.
(Ed.), Oxford Bibliographies in “Anthropology.” Oxford University Press,
New York.

</div>

<div id="ref-r_core_team_r_2019" class="csl-entry">

Team, R.C., 2019. R: A language and environment for statistical
computing. R Foundation for Statistical Computing, Vienna, Austria.

</div>

<div id="ref-rstudio_team_rstudio_2019" class="csl-entry">

Team, Rs., 2019. RStudio: Integrated development for r. RStudio, Inc.,
Boston, MA.

</div>

<div id="ref-white_ethno-archaeology_1967" class="csl-entry">

White, J.P., 1967. Ethno-archaeology in new guinea: Two examples.
Mankind 6, 409–414.

</div>

</div>
