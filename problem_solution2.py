import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plots
sns.set(style="whitegrid")

# Load the datasets
timeseries_data = pd.read_parquet('./timeseries.parquet')
occupancy_data = pd.read_parquet('./occupancy.parquet')

# Define the required fields dynamically
fields = timeseries_data['field'].unique()
temp_sensor_field = next((field for field in fields if "temperature_sensor" in field), None)
temp_setpoint_field = next((field for field in fields if "temperature_setpoint" in field), 'effective_cooling_zone_air_temperature_setpoint')

# Problem 1: Analyze Temperature Maintenance
if temp_sensor_field and temp_setpoint_field:
    temperature_data = timeseries_data[timeseries_data['field'].isin([temp_sensor_field, temp_setpoint_field])]
    temperature_data['value'] = pd.to_numeric(temperature_data['value'], errors='coerce')
    temperature_data_pivot = temperature_data.pivot_table(index='date_time_local', columns='field', values='value', aggfunc='first')
    temperature_data_pivot.ffill(inplace=True)
    temperature_data_pivot['temp_deviation'] = temperature_data_pivot[temp_setpoint_field] - temperature_data_pivot[temp_sensor_field]
    
    # Plotting temperature deviation
    plt.figure(figsize=(14, 7))
    plt.plot(temperature_data_pivot.index, temperature_data_pivot['temp_deviation'], color='deepskyblue', label='Temperature Deviation')
    plt.title('Temperature Deviation Over Time', fontsize=16)
    plt.ylabel('Temperature Deviation (°C)', fontsize=14)
    plt.xlabel('Date and Time', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# Problem 2: Time to Reach Setpoint
if 'temperature_data_pivot' in locals() and not temperature_data_pivot.empty:
    occupied_data = timeseries_data[timeseries_data['field'] == 'occupied_mode']
    occupied_data['value'] = pd.to_numeric(occupied_data['value'], errors='coerce')
    occupied_transitions = occupied_data[occupied_data['value'].diff() != 0]
    occupied_transitions.set_index('date_time_local', inplace=True)
    temperature_transitions = temperature_data_pivot.reindex(occupied_transitions.index, method='nearest')
    temperature_transitions['time_to_reach'] = temperature_transitions.index.to_series().diff().dt.total_seconds()
    print("Average Time to Reach Setpoint:", temperature_transitions['time_to_reach'].dropna().mean(), "seconds")

# Problem 3: Unified Occupancy Metric
try:
    weights = {'people_in': 0.5, 'traffic': 0.3, 'occupancy': 0.2}
    for key in weights:
        if key in occupancy_data.columns:
            occupancy_data[key] *= weights[key]
    occupancy_data['unified_metric'] = occupancy_data[['people_in', 'traffic', 'occupancy']].sum(axis=1)
    print(occupancy_data[['organization_id', 'building_name', 'space_name', 'unified_metric']].head())
except KeyError as e:
    print("KeyError in Problem 3:", e)

# Problem 4: Occupancy Dashboard
plt.figure(figsize=(18, 12))
plt.subplot(2, 1, 1)
sns.lineplot(data=occupancy_data, x='date_time', y='unified_metric', hue='building_name', palette='viridis')
plt.title('Unified Occupancy Metric Over Time', fontsize=16)
plt.xlabel('Date and Time', fontsize=14)
plt.ylabel('Unified Metric', fontsize=14)
plt.legend(title='Building Name')

plt.subplot(2, 1, 2)
sns.boxplot(data=occupancy_data, x='building_name', y='unified_metric', palette='magma')
plt.title('Occupancy Distribution by Building', fontsize=16)
plt.xlabel('Building Name', fontsize=14)
plt.ylabel('Unified Metric', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.show()
