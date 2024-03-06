author
------

Romeo De Timmerman


project summary
---------------

This repository contains a quantitative analysis of blues lyrics performed by artists across time and socio-cultural groups. It specifically trains a gradient boosting classifier using the Catboost library to predict the use of certain features of African American English in a corpus of 270 studio recorded songs. This analysis is a part of [my PhD project](https://research.flw.ugent.be/en/projects/indexical-expressions-authenticity-mixed-method-inquiry-stylistic-use-african-american) on the use of African American English features as indexical expressions of authenticity in blues music. 


project structure
-----------------

```
PROJECT DIRECTORY
    │
    ├── README.md          <- general information on the project
    │
    ├── CITATION           <- information on how to cite this project
    │
    ├── LICENSE            <- information on the copyright license of this project
    │
    ├── data
    │   ├── raw            <- original, immutable datasets
    │   ├── interim        <- intermediate, pre-processed datasets
    │   └── processed      <- final, processed datasets 
    │
    ├── models             <- any exported machine learning models
    │
    ├── notebooks          <- all jupyter notebooks using number-initials-title naming convention,
    │                         e.g. "1-rdt-initial-data-exploration"
    │
    ├── reports            <- exported jupyter notebooks in HTML or PDF format
    │   └── figures        <- exported graphics and figures
    │
    ├── environment.yml    <- YAML file to build conda environment used in this project
    │
    └── src                <- Python source code

(this project structure is based on the Cookiecutter Data Science template)
```