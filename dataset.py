import os
import sys
import traci
import csv
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

if "SUMO_HOME" not in os.environ:
    sys.exit(" Error: Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

sumo_config = [
    "sumo-gui",
    "-c", "extra.sumocfg",
    "--step-length", "1"
]

traci.start(sumo_config)


def calculate_congestion(vehicle_count, max_capacity=15):
    return round(min(vehicle_count / max_capacity, 1), 2)


csv_filename = "traffic_data.csv"
csv_header = ["timestamp", "road_id", "vehicle_count", "avg_speed", "congestion_level"]

if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

print(" Running SUMO Simulation... Saving Traffic Data in CSV")

start_time = time.time()

data_list = []

base_time = datetime.now()

for step in range(300):
    traci.simulationStep()

    sim_time_seconds = traci.simulation.getTime()
    current_time = base_time + timedelta(seconds=sim_time_seconds)
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Simulation Step: {step + 1} | Time: {formatted_time}")

    for edge in traci.edge.getIDList():
        if edge.startswith(":"):
            continue

        vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
        speed_avg = traci.edge.getLastStepMeanSpeed(edge)
        congestion_level = calculate_congestion(vehicle_count) if vehicle_count > 0 else 0

        data_list.append([formatted_time, edge, vehicle_count, speed_avg, congestion_level])

        print(
            f" Road: {edge} | Vehicles: {vehicle_count} | Avg Speed: {speed_avg:.2f} m/s | Congestion: {congestion_level}")

with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data_list)

print(" SUMO Simulation Complete. Data Saved to traffic_data.csv ")

end_time = time.time()
execution_time = round(end_time - start_time, 2)
print(f" Execution Time: {execution_time} seconds")

print("Preprocessing Traffic Data for LSTM Training...\n")

df = pd.DataFrame(data_list, columns=["timestamp", "road_id", "vehicle_count", "avg_speed", "congestion_level"])

if df.empty:
    print(" Error: No data collected! Preprocessing failed.")
    sys.exit(1)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["time_seconds"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()

df.drop(columns=["timestamp"], inplace=True)

scaler = MinMaxScaler()
df[["vehicle_count", "avg_speed", "congestion_level", "time_seconds"]] = scaler.fit_transform(
    df[["vehicle_count", "avg_speed", "congestion_level", "time_seconds"]]
)

print(" Preprocessed Data Sample:\n", df.head())

preprocessed_csv = "preprocessed_traffic_data.csv"
df.to_csv(preprocessed_csv, index=False)

print(f"Preprocessed data saved to {preprocessed_csv}")
traci.close()