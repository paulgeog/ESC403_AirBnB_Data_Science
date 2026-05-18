# ESC403_AirBnB_Data_Science

## Overview
The short-term rental market, AirBnB in particular, has grown rapidly in urban areas, influencing rents, local economies, and urban planning. In cities like Zurich, understanding the factors of Airbnb prices can provide insights into market dynamics, potential regulatory interventions, and correlations with traditional rental prices. In our project we aim to analyse Airbnb listings in Zürich, identify key features influencing pricing, and compare them with neighbourhood-level rental statistics.
The goal of this project is to explore the following:
- EDA
- price prediction based on key predictors within the dataset
- comparison to rental prices in Zurich's districts
- other machine learning processes to cluster and categorize the listings in the dataset based on review scores etc.
***
## Datasets
#### Zurich Airbnb listings: *'listings.csv'*
- *Source*: Inside AirBnB: https://insideairbnb.com/get-the-data/
- *Features*: 3'417 (After cleaning: 2'592)
- *Variables*: 78 (After cleaning: 71+30)
#### Housing stock Zurich: *'bau522od5221_wohnungsbestand_zurich.csv'*
- *Source*: Inside AirBnB: https://data.stadt-zuerich.ch/dataset/bau_whg_bestand_ea_zizahl_quartier_seit2010_od5221 
- *Features*: 15'978 (After cleaning: 1002)
- *Variables*: 15 (After cleaning: 14)
#### Zurich Rental Prices per Neighbourhood: *'rental_prices.csv'*
- *Source*: Inside AirBnB: https://data.stadt-zuerich.ch/dataset/bau_whg_mpe_mietpreis_raum_zizahl_gn_jahr_od5161
- *Features*: 2'632 (After cleaning: 68)
- *Variables*: 35 (After cleaning: 29)
#### Zurich quartiere boundaries: *'zurich_quartiere.gpkg'*
- *Source*: Stadt Zürich: https://www.stadt-zuerich.ch/geodaten/download/Statistische_Quartiere?format=10005 
#### Zurich forest boundaries: *'forest_zurich.gpkg'*
- *Source*: swisstopo: https://www.swisstopo.admin.ch/en/national-map-swiss-map-vector-25 
#### Zurich lake boundaries: *'zurichsee.gpkg'*
- *Source*: swisstopo: https://www.swisstopo.admin.ch/en/national-map-swiss-map-vector-25 
***
## Project Structure
``` 
├── data/                       # Original datasets and quartiere/forest boundaries
├── airbnb_analysis.ipynb       # main notebook of our analysis workflow
├── filter_datasets.ipynb       # explanation of normalization workflow
├── src/                        # all .py files with code for airbnb_analysis.ipynb
├── README.md                   # this file
└── requirements.txt            # all dependencies needed for running this project
```
### Execution order
Running `airbnb_analysis.ipynb`is enough to get the entire analysis. Data normalization and other helper steps are handled in this singular file.
## Dependencies
All required dependencies to run the code in the jupyter notebooks and the .py files in the src folder are listed in requirements.txt
Python version 3.10
