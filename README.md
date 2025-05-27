# Air Quality Analysis Prototype

This repository contains the source code for a Python-based software prototype developed for the comprehensive analysis of air quality data. The system is designed to automate the workflow from data ingestion and cleaning to advanced time-series analysis and visualization.

The prototype was developed as part of a diploma thesis to demonstrate a practical application of data analysis techniques in environmental science. The included case study analyzes PM2.5 and PM10 concentration data from the "Ventspils, parventa" monitoring station in Latvia.

**CSV file was downloaded from open data websource OpenAQ, time interval of measurements 01.04.25 - 30.04.25**

**Features**
- **CSV Data Ingestion**: Loads time-series data from a structured CSV file.
- **Data Cleaning & Preprocessing**:  
  - Automatically handles data type conversion  
  - Removes invalid or anomalous entries (e.g., negative concentrations)  
  - Restructures data for analysis
- **Descriptive Statistics**: Calculates key statistical metrics (mean, median, standard deviation, etc.) for each pollutant.
- **Time-Series Analysis**:  
  - **Moving Averages**: Calculates rolling averages to identify underlying trends.  
  - **STL Decomposition**: Decomposes time series into trend, seasonal, and residual components to uncover cyclical patterns.  
  - **Correlation Analysis**: Calculates and visualizes the Pearson correlation between pollutants (e.g., PM2.5 vs. PM10).
- **Rich Visualizations**: Generates a suite of plots using Matplotlib and Plotly, including:  
  - Time-series plots  
  - Histograms  
  - Box plots  
  - Scatter plots with regression lines
- **Spatial Mapping**: Creates an interactive HTML map using Folium to visualize the geographic location of the data source.
