# Route Rationalization Model Using Machine Learning & AI for Real-Time Traffic Management

## Overview

An intelligent traffic prediction system that combines SUMO traffic simulation with deep learning to forecast traffic conditions. The system enables proactive traffic management by predicting future congestion levels before they occur.

## Problem Statement

Urban traffic congestion presents significant challenges to modern cities:

- **Economic Impact**: Billions lost annually in productivity and fuel costs
- **Environmental Concerns**: Increased emissions and air pollution
- **Public Safety**: Delayed emergency response times and higher accident risks
- **Quality of Life**: Longer commute times and reduced mobility
- **Infrastructure Strain**: Accelerated road wear and maintenance costs

Current traffic management systems are primarily reactive, addressing congestion only after it occurs. This approach is inefficient and costly. Our solution shifts to a proactive model, predicting and preventing traffic issues before they impact the city's operations.

## Solution Features

### 1. Traffic Simulation

- SUMO-based traffic modeling
- Multi-road network simulation
- Data collection (vehicle counts, speeds, congestion levels)

### 2. LSTM Model

- Three-layer architecture (128, 64, 32 units)
- Batch normalization and dropout (0.2)
- 5-step lookback prediction
- Early stopping optimization

### 3. Data Processing

- Efficient preprocessing
- MinMaxScaler normalization
- Multi-feature handling
- Time-series analysis

### 4. Prediction System

- 5-step ahead forecasting
- Multi-road predictions
- Real-time updates
- Tabulated output

## Key Benefits

### For Authorities

- Proactive traffic control
- Resource optimization
- Emergency planning
- Data-driven decisions

### For Public

- Reduced travel times
- Better route planning
- Real-time updates
- Environmental benefits

### For Infrastructure

- Better maintenance planning
- Optimized usage
- Emergency access
- Reduced wear

## Project Structure

```
.
├── extra_model.py              # LSTM model
├── extra_prediction.py         # Prediction logic
├── dataset.py                  # Data handling
├── extra.sumocfg              # SUMO config
├── traffic_lstm_model_optimized.h5  # Trained model
├── preprocessed_traffic_data.csv    # Processed data
├── traffic_data.csv           # Raw data
├── extra.netecfg             # Network config
├── extra.rou.xml             # Route definitions
└── extra.net.xml             # Network definition

```

## Dependencies

- Python 3.9+
- SUMO (Simulation of Urban MObility) 1.15.0
- TensorFlow 2.12.0
- NumPy 1.24.3
- Pandas 2.0.3
- scikit-learn 1.3.0
- Matplotlib 3.7.2
- tabulate 0.9.0

## Installation and Usage

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow==2.12.0 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.2 tabulate==0.9.0

# Install SUMO (Windows)
# Download and install from: https://sumo.dlr.de/docs/Installing/Windows_Build.html
```

### 2. Project Workflow

#### Step 1: Data Collection

```bash
# Run SUMO simulation and collect traffic data
python dataset.py
# This will generate traffic_data.csv
```

![alt text](extra_data_prediction.png)

#### Step 2: Model Training

```bash
# Train the LSTM model
python extra_model.py
# This will:
# - Process traffic_data.csv
# - Train the model
# - Save as traffic_lstm_model_optimized.h5
```

![alt text](<Screenshot 2025-03-21 234458.png>)

#### Step 3: Traffic Prediction

```bash
# Run predictions
python extra_prediction.py
# This will:
# - Load the trained model
# - Process the data
# - Display predictions in a table format
```

![alt text](<extra predicted .png>)

### 3. Configuration

- Ensure SUMO is in your system PATH
- The simulation parameters can be adjusted in `extra.sumocfg`
- Network configuration is in `extra.net.xml`
- Route definitions are in `extra.rou.xml`

### 4. Expected Output

- `traffic_data.csv`: Raw traffic simulation data
- `preprocessed_traffic_data.csv`: Processed data for model training
- `traffic_lstm_model_optimized.h5`: Trained model
- Prediction results displayed in a formatted table

## Applications

1. **Traffic Management**

   - Dynamic signal control
   - Speed limit adjustments
   - Lane management

2. **Urban Planning**

   - Infrastructure decisions
   - Transport optimization
   - Emergency routes

3. **Public Services**
   - Real-time updates
   - Route recommendations
   - Emergency response

## Target Users

- Traffic management departments
- Emergency services
- Transportation authorities
- Urban planners
- General public

## Team Members

### Aman Patre

- Led the development of the LSTM model architecture
- Implemented model training and optimization
- Developed the prediction system
- Handled data preprocessing and feature engineering

### Ashutosh Yadav

- Designed and implemented the SUMO traffic simulation
- Created the network configuration and route definitions
- Developed the data collection pipeline
- Implemented real-time traffic monitoring

### Priyanshu Bidhuri

- Developed the data processing pipeline
- Implemented the visualization components
- Created the prediction output formatting
- Handled model evaluation and performance metrics

### Soumya Yadav

- Implemented the traffic simulation scenarios
- Developed the congestion level calculation
- Created the data validation system
- Handled the integration testing

## Technical Innovation

- Simulation + ML integration
- Advanced LSTM architecture
- Multi-point processing
- Actionable insights

## Future Enhancements

### 1. Dynamic Route Optimization

- Implementation of A\* algorithm for optimal path finding
- Real-time vehicle rerouting based on predicted congestion levels
- Integration with traffic signal optimization
- Dynamic speed limit adjustments

### 2. Real-Time Map Integration

- Integration with real-time mapping services
- Live traffic visualization
- Interactive route planning interface
- Real-time congestion alerts and alternative route suggestions

### 3. System Improvements

- Enhanced prediction accuracy through continuous learning
- Expanded road network coverage
- Integration with smart city infrastructure
- Mobile application for end-user access

These enhancements will transform the system from a prediction tool into a comprehensive traffic management solution, providing real-time optimization and user guidance.
