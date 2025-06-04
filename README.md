# Replication Code for "Covering Blue Voices: African American English and Authenticity in Blues Covers"

## author

Romeo De Timmerman

## project summary

This repository contains data for a quantitative analysis of blues lyrics performed by artists across time and socio-cultural groups. This analysis is a part of my PhD project on the use of African American English features as indexical expressions of authenticity in blues music. The particular study for which data is shared here examines the use of African American English (AAE) features in blues music, for which a corpus of 270 studio-performed blues songs was compiled from YouTube, consisting of six songs each by 45 artists. These artists were evenly distributed across three social groups (African American; non-African American, US-based; and non-African American, non-US-based) and three time periods (the 1960s, 1980s, and 2010s). Each artist contributed three original songs and three covers (i.e., previously recorded by other performers). Songs were selected to fit broad blues criteria, including structural, melodic, and lyrical patterns, encompassing traditional blues and contemporary blues-rock. All 270 songs were imported into MAXQDA for transcription and annotation of five phonological and three lexico-grammatical AAE features, selected based on established sociolinguistic literature. Each token where a feature could potentially occur was coded in binary fashion (realized or not), with uncertain cases left uncoded. The annotated data were exported from MAXQDA into a structured tabular format for statistical and machine learning analysis in Python. Only the Python code used to (pre)process and analyze the data are included in this repository. The raw, intermediate and processed datasets can be accessed via this TROLLing repository: https://doi.org/10.18710/DOJXAV

## project structure

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documents
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-rdt-initial-data-exploration`.
│
├── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py

(this project structure is based on the Cookiecutter Data Science template)
```
