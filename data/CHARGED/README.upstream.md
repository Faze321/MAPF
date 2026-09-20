# CHARGED Dataset Description

## Overview

The CHARGED (City-scale and Harmonized Dataset for Global Electric Vehicle Charging Demand Analysis) dataset provides comprehensive electric vehicle charging data and related auxiliary information for six major cities worldwide: **Los Angeles**, **São Paulo**, **Shenzhen**, **Amsterdam**, **Johannesburg**, and **Melbourne**.

## Dataset Versions

Each city has two data versions:
- **Standard version**: Complete dataset with all charging sites
- **`xxx_remove_zero` version**: Filtered dataset with charging sites that have zero charge volume, charge duration, and price removed for cleaner analysis

## Data Structure

Each city folder contains 10 CSV files. Due to size limitations, some files larger than 100MB are compressed in `csv-larger-than-100MB.rar`.

## File Descriptions

### Core Charging Data

#### volume.csv
- **Description**: Hourly charging volume (energy consumption) data
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Site IDs
- **Unit**: Kilowatt-hours (kWh)
- **Use Case**: Primary target variable for demand prediction

#### duration.csv
- **Description**: Charging duration data representing time spent charging
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Site IDs
- **Unit**: Hours (h)
- **Note**: Values represent charging duration between consecutive time intervals

### Pricing Information

#### e_price.csv
- **Description**: Electricity charging fees (energy cost)
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Site IDs
- **Units by City**:
  - Los Angeles: USD/kWh
  - São Paulo: Brazilian Real (BRL)/kWh
  - Shenzhen: Chinese Yuan (CNY)/kWh
  - Amsterdam: Dutch Guilder/kWh
  - Johannesburg: South African Rand (ZAR)/kWh
  - Melbourne: Australian Dollar (AUD)/kWh

#### s_price.csv
- **Description**: Service fees (additional charges beyond electricity cost)
- **Format**: Time series with hourly granularity
- **Index**: Timestamps (hourly intervals)
- **Columns**: Site IDs
- **Note**: São Paulo, Johannesburg, and Melbourne have no service fees (uniformly set to 0)
- **Units**: Same as e_price.csv for each respective city

### Infrastructure Information

#### sites.csv
- **Description**: Comprehensive information about charging sites
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| site_id | Unique identifier for the charging site | N/A |
| longitude | Geographical longitude of site location | Degrees |
| latitude | Geographical latitude of site location | Degrees |
| charger_num | Number of charging chargers at the site | Count |
| total_duration | Total charging duration recorded at site | Hours |
| total_volume | Total charging volume recorded at site | kWh |
| avg_power | Average charging power per charging record | kW |
| perimeter | Perimeter of DBSCAN cluster shape | Meters (m) |
| area | Area of DBSCAN cluster shape | Square meters (m²) |

#### chargers.csv
- **Description**: Detailed information about individual charging chargers
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| charger_id | Unique identifier for the charging charger | N/A |
| longitude | Geographical longitude of charger location | Degrees |
| latitude | Geographical latitude of charger location | Degrees |
| site_id | Identifier of the site containing this charger | N/A |
| total_duration | Total charging duration at this charger | Hours |
| total_volume | Total energy delivered at this charger | kWh |
| avg_power | Average charging power per charging record | kW |

### Auxiliary Data

#### distance.csv
- **Description**: Inter-site distance matrix
- **Format**: Symmetric matrix
- **Index/Columns**: Site IDs
- **Calculation**: Geodesic distance based on ellipsoidal Earth model
- **Unit**: Kilometers (km)
- **Use Case**: Spatial analysis and clustering

#### weather.csv
- **Description**: Hourly weather data from Visual Crossing API
- **Source**: [Visual Crossing Weather API](https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/)
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| temp | Temperature | Celsius (°C) |
| feelslike | Apparent temperature (heat index/wind chill) | Celsius (°C) |
| humidity | Relative humidity | Percentage (%) |
| dew | Dew point temperature | Celsius (°C) |
| precip | Liquid precipitation amount | mm |
| snow | Snowfall amount | mm |
| snowdepth | Snow depth on ground | mm |
| preciptype | Precipitation type | Array (rain/snow/freezingrain/ice) |
| windgust | Instantaneous wind speed | km/h |
| windspeed | Sustained wind speed | km/h |
| winddir | Wind direction | Degrees |
| pressure | Sea level atmospheric pressure | hPa |
| visibility | Visibility distance | km |
| cloudcover | Cloud coverage | Percentage (0-100%) |
| solarradiation | Solar radiation power | W/m² |
| solarenergy | Solar energy accumulation | MJ/m² |
| uvindex | UV exposure index | Scale (0-10) |
| conditions | Weather conditions | Categorical (see mapping below) |

**Weather Conditions Mapping**:
- 0: Clear
- 1: Overcast
- 2: Partially cloudy
- 3: Rain
- 4: Rain, Fog
- 5: Rain, Overcast
- 6: Rain, Partially cloudy
- 7: Snow
- 8: Snow, Fog
- 9: Snow, Partially cloudy
- 10: Snow, Rain
- 11: Snow, Rain, Overcast
- 12: Snow, Rain, Partially cloudy

#### poi.csv
- **Description**: Points of Interest (POIs) from OpenStreetMap
- **Source**: [OpenStreetMap](https://www.openstreetmap.org/)
- **Categories**: Amenities, buildings, offices, shops, leisure facilities, etc.
- **Reference**: [OSM Wiki - Map Features](https://wiki.openstreetmap.org/wiki/Map_features)
- **Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| type | Geographic feature category | N/A |
| longitude | POI longitude | Degrees |
| latitude | POI latitude | Degrees |

### Metadata

#### info.csv
- **Description**: Dataset metadata and summary statistics
- **Key Fields**:

| Field | Description | Unit |
|-------|-------------|------|
| city | City name | N/A |
| country | Country name | N/A |
| abbreviation | City code (3-letter) | N/A |
| total_chargers | Total number of charging chargers | Count |
| total_sites | Total number of charging sites | Count |
| DBSCAN_eps | DBSCAN clustering radius parameter | Meters (m) |
| total_duration | Total charging duration in dataset | Hours |
| total_volume | Total charging volume in dataset | kWh |
| avg_power | Average charging power across all records | kW |

## Data Quality Notes

### Preprocessing Applied
- **Anomaly Detection**: Outliers identified and corrected using IQR method
- **Zero Sequence Handling**: Zero sequences interpolated
- **Site Clustering**: Geographic clustering using DBSCAN algorithm

### Known Limitations
- Some cities may have missing data during certain time periods
- Weather data availability depends on API coverage
- POI data completeness varies by city and OSM coverage
- Currency exchange rates not provided for cross-city comparisons

## Support

For questions about the dataset, please refer to:
- Main project documentation: [README.md](../README.md)
- GitHub Issues: [Project Issues](https://github.com/IntelligentSystemsLab/CHARGED/issues)
- Contact: guozh29@mail2.sysu.edu.cn

