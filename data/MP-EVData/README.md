##### **MP-EVData: An AI-Augmented Dataset of Multi-Prototype Electric Vehicle Charging Load Profiles in China​**

This repository contains the dataset and corresponding code for the paper "MP-EVData: An AI-Augmented Dataset of Multi-Prototype Electric Vehicle Charging Load Profiles in China", which has been submitted to Nature Scientific Data. The resources provided here aim to facilitate further research and analysis in the field of electric vehicle charging load profiling.​

###### **Dataset Overview​**

The dataset folder includes four xlsx files, each capturing different aspects of electric vehicle charging load data across ten charging stations in China:​

**charging session.xlsx**: This file contains detailed charging session data from the ten stations. It includes information related to individual charging events, providing a granular view of charging activities.​

**station-level load Profile 15min.xlsx**: It presents the daily load profiles of the ten stations with a time resolution of 15 minutes. This high-resolution data allows for in-depth analysis of short-term load fluctuations.​

**station-level load Profile 1h.xlsx**: This file offers the daily load profiles of the ten stations at an hourly resolution, suitable for analyzing longer-term load patterns.​

**price.xlsx**: It contains time-of-use electricity price data for the ten stations, which is crucial for understanding the economic aspects of electric vehicle charging.​

###### **Code Structure​**

The code folder consists of subfolders and scripts that support the data processing, analysis, and validation described in the paper:​

**forecast subfolder**: This subfolder corresponds to the "Application to Load Forecasting" section in the Technical Validation part of the paper. It leverages methods and references from the GitHub repository: https://github.com/KimMeen/Time-LLM, which has been appropriately cited in the paper.​

**generation subfolder**: It is related to the "AI-Augmented Synthetic Data Generation" section in the Methods part of the paper. The code here refers to the GitHub repository: https://github.com/LSY-Cython/DiffCharge, with proper citations in the paper.​

**data_process_charging.py and data_process_swaping.py**: These scripts are associated with the "Data Preprocessing and Station-Level Load Aggregation" section in the Methods part. They handle the preprocessing of raw data and the aggregation of load data at the station level.​

**plot_XX.py scripts**: These scripts are responsible for generating the various figures presented in the paper. Each script corresponds to specific figures, aiding in the visualization and interpretation of the data and results.​

**calculate_stats.py scripts**：This script is used to compute the quantitative data in Table 3.
By providing this dataset and code, we hope to contribute to the advancement of research on electric vehicle charging load profiles and related applications. For any questions or further information, please refer to the paper or contact the corresponding authors.​



