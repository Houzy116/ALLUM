# Satellites reveal recent declines in global land surface albedo-induced radiative forcing

This repository contains the codes and the intermediate data for the study entitled **"Satellites reveal recent declines in global land surface albedo-induced radiative forcing"**. 

## Repository Structure

The repository is organized into the following directories and files:

- **data/**: Contains the intermediate data used in the study.
- **figure/**: Stores the figures can be recreated using the scripts in the repository.
- **preprocess/**: Contains scripts for data preprocessing.
- **grid_data/**: Includes scripts for generating grid data in four ALLUMs.
- **val/**: Contains scripts for result validation.
- **anasys/**: Contains scripts for analysis.
- **tool.py**: A script that includes common utility functions and variables used in the repository.



<p align="center">
  <img src="https://github.com/Houzy116/ALLUM/blob/master/figure/S1.png" alt="S1" width="600">
   <br>
  <em><strong>Fig.1 The framework of our methodology. a,</strong> Generation of four ALLUMs, and contributions of LULC change to global snow-free land surface mean albedo (GLMA) change and the induced radiative forcing. <strong>b,</strong> Validations for reconstructing 500m-resolution blue-sky albedo. <strong>c,</strong> Contributions of changes in photosynthetic vegetation (PV), non-photosynthetic vegetation (NPV) and surface water content (SWC) to albedo change over regions without LULC conversions. The data in pink boxes and yellow boxes separately represent 500m-resolution pixel-level data and grid-level data.</em>
</p>

# Dependency Installation Guide

Follow these steps to install the dependencies for this GitHub repository:


## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Houzy116/ALLUM.git
   cd ALLUM
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
##  Codes have been tested on

### Python Versions
- **Python 3.7.7**

### Operating Systems
- **Ubuntu 18.04**

### Specific Package Versions
- **astropy 4.3.1**
- **GDAL 3.0.2**
- **geopandas 0.8.1**
- **matplotlib 3.5.3**
- **mpl_chord_diagram 0.4.1**
- **mpl_scatter_density 0.7**
- **netCDF4 1.5.4**
- **numpy 1.19.1**
- **pandas 1.3.5**
- **pymannkendall 1.4.3**
- **python_dateutil 2.8.1**
- **python_ternary 1.0.8**
- **rasterio 1.1.5**
- **scikit_learn 0.24.1**
- **scipy 1.5.2**
- **seaborn 0.11.0**
- **Shapely 1.7.1**
- **squarify 0.4.3**
- **ternary 0.1**
- **torch 1.12.0+cu113**
- **tqdm 4.50.2**
- **pyhdf 0.11.4**
- **joblib 0.17.0**






