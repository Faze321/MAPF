import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_power_data_from_csv(file_path):

    df = pd.read_csv(file_path)
    return df['power'].values

def generate_tou_price():
    """
    Generate Time-of-Use (TOU) price curve.
    """
    price = np.zeros(96)
    price[76:84] = 1.36  # Sharp Peak price (19:00 - 21:00)
    price[32:44] = 1.11  # Peak price (8:00 - 11:00)
    price[72:76] = 1.11  # Peak price (18:00 - 19:00)
    price[24:32] = 0.67  # Shoulder price (06:00 - 8:00)
    price[44:72] = 0.67  # Shoulder price (11:00 - 18:00)
    price[84:88] = 0.67  # Shoulder price (21:00 - 22:00)
    price[88:] = 0.33  # Off-peak price (22:00 - 00:00)
    price[:24] = 0.33  # Off-peak price (00:00 - 06:00)
    return price

def draw_tou_impact_plot(load_1, label_1, load_2, label_2, tou_price):
    """
    Draw and save the graph showing two load curves and one price curve.
    """
    fig = plt.figure(dpi=150)
    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    l1, = ax1.plot(load_1, label=label_1, color="royalblue")
    l2, = ax1.plot(load_2, label=label_2, linestyle='--', color="darkorange")
    l3, = ax2.step(range(96), tou_price, label="TOU Price", color="slategray", where='post', alpha=0.7)

    fs = 14  # Font size
    ax1.set_ylabel("Power(kW)", fontsize=fs)
    ax1.set_xlabel("Time(h)", fontsize=fs)
    ax2.set_ylabel("Price(RMB/kWh)", fontsize=fs)

    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)

    ax1.set_xlim(0, 95)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    lns = [l1, l2, l3]
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, fontsize=12, loc='upper left')

    ax1.tick_params(axis='both', direction='in', labelsize=12)
    ax2.tick_params(axis='y', direction='in', labelsize=12)
    positions = list(range(0, 96 + 1, 12))
    ticks = list(range(0, 24 + 1, 3))
    plt.xticks(positions, ticks)

    output_filename = "output/tou_impact_on_two_taxis_final.png"
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig("output/tou_impact_on_two_taxis.png")
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    taxi_station_1_path = 'A1.csv'
    taxi_station_2_path = 'A2.csv'

    station_1_load = load_power_data_from_csv(taxi_station_1_path)
    station_2_load = load_power_data_from_csv(taxi_station_2_path)

    tou_price_curve = generate_tou_price()

    if station_1_load is not None and station_2_load is not None:
        # Pass new labels to update legend
        draw_tou_impact_plot(
            station_1_load, "Taxi A1",
            station_2_load, "Taxi A2",
            tou_price_curve
        )
    else:
        print("error")