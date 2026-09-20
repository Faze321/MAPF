import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from collections import defaultdict


def process_new_dataset(input_file, output_folder, granularity='15min'):
    """
    :param granularity: (15min or 1h）
    """
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_file)
    df['Connection_Time'] = pd.to_datetime(df['Connection_Time'])
    df['Disconnection_Time'] = pd.to_datetime(df['Disconnection_Time'])
    df['Charged_Energy'] = pd.to_numeric(df['Charged_Energy'], errors='coerce')

    if granularity == '15min':
        time_points = 96
        time_seconds = 900
        freq = '15min'
    elif granularity == '1h':
        time_points = 24
        time_seconds = 3600
        freq = 'H'
    else:
        raise ValueError("please choose '15min' or '1h'")

    station_data = defaultdict(lambda: np.zeros(time_points))

    for _, row in df.iterrows():
        start_time = row['Connection_Time']
        end_time = row['Disconnection_Time']
        energy = row['Charged_Energy']  # kWh

        if granularity == '15min':
            start_idx = (start_time.hour * 4) + (start_time.minute // 15)
            end_idx = (end_time.hour * 4) + (end_time.minute // 15)
        else:
            start_idx = start_time.hour
            end_idx = end_time.hour

        total_duration = (end_time - start_time).total_seconds() / time_seconds
        if total_duration <= 0:
            continue

        unit_power = energy * (time_points / 24) / total_duration

        current_time = start_time
        while current_time < end_time:
            date_key = current_time.strftime("%Y-%m-%d")

            day_start = pd.Timestamp(date_key)
            if granularity == '15min':
                current_idx = (current_time.hour * 4) + (current_time.minute // 15)
            else:
                current_idx = current_time.hour

            next_time = min(
                day_start + timedelta(days=1),
                current_time + timedelta(seconds=time_seconds),
                end_time
            )

            duration = (next_time - current_time).total_seconds() / time_seconds
            station_data[date_key][current_idx] += unit_power * duration

            current_time = next_time

    rows = []
    sorted_dates = sorted(station_data.keys())

    for date in sorted_dates:

        daily_power = station_data[date]

        timestamps = pd.date_range(
            start=date,
            periods=time_points,
            freq=freq
        )

        for ts, power in zip(timestamps, daily_power):
            rows.append({
                'datetime': ts.strftime("%Y-%m-%d %H:%M"),
                'power': round(power,1)
            })

    if rows:
        output_path = os.path.join(output_folder, f'power_{granularity}.csv')
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Generation successful：{output_path}")
    else:
        print("No valid data generated")

if __name__ == "__main__":
    process_new_dataset("A10.csv", "output", granularity='1h') #A1,A2,...,A10
