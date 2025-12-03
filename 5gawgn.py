import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

#BER
def ber_awgn(mod, snr_db):
    snr = 10 ** (snr_db / 10)

    if mod == "BPSK":
        return 0.5 * erfc(np.sqrt(snr))

    elif mod == "QPSK":
        return 0.5 * erfc(np.sqrt(snr / 2))

    elif mod == "8-PSK":
        return erfc(np.sqrt(snr) * np.sin(np.pi / 8))

    elif mod == "16-QAM":
        return (3/8) * erfc(np.sqrt((4/10) * snr))

    elif mod == "64-QAM":
        return (7/24) * erfc(np.sqrt((12/42) * snr))

    elif mod == "256-QAM":
        return (15/64) * erfc(np.sqrt((32/170) * snr))

    return np.zeros_like(snr)


#Simulation

def simulate_ber_qos(modulation, snr_db):

    ber = ber_awgn(modulation, snr_db)

    # QoS parameters
    Rb = 1e6
    BW = 1e6

    throughput = (1 - ber) * Rb / 1e6
    delay = (1 / (throughput * 1e6)) * 1e3
    spectral_eff = Rb / BW * (1 - ber)

    metrics = {
        "Avg BER": np.mean(ber),
        "Throughput (Mbps)": np.mean(throughput),
        "Delay (ms)": np.mean(delay),
        "Spectral Eff (bits/s/Hz)": np.mean(spectral_eff)
    }

    return ber, metrics


#gui

class ChannelSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Channel Simulation")
        self.root.geometry("1000x700")

        self.curves_data = []

        control_frame = ttk.LabelFrame(root, text="Simulation Controls")
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Modulation:").grid(row=0, column=0, padx=5)
        self.mod_box = ttk.Combobox(
            control_frame,
            values=["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "256-QAM"],
            width=12
        )
        self.mod_box.set("BPSK")
        self.mod_box.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="name:").grid(row=0, column=2, padx=5)
        self.curve_name = ttk.Entry(control_frame, width=15)
        self.curve_name.insert(0, "Curve")
        self.curve_name.grid(row=0, column=3, padx=5)

        ttk.Label(control_frame, text="SNR Start (dB):").grid(row=1, column=0, padx=5)
        self.snr_start = ttk.Entry(control_frame, width=10)
        self.snr_start.insert(0, "0")
        self.snr_start.grid(row=1, column=1, padx=5)

        ttk.Label(control_frame, text="SNR End (dB):").grid(row=1, column=2, padx=5)
        self.snr_end = ttk.Entry(control_frame, width=10)
        self.snr_end.insert(0, "5")
        self.snr_end.grid(row=1, column=3, padx=5)

        ttk.Label(control_frame, text="Points:").grid(row=1, column=4, padx=5)
        self.points = ttk.Entry(control_frame, width=10)
        self.points.insert(0, "5")
        self.points.grid(row=1, column=5, padx=5)

        ttk.Button(control_frame, text="Plot", command=self.add_curve).grid(row=0, column=6, rowspan=2, padx=10)
        ttk.Button(control_frame, text="Clear", command=self.clear_curves).grid(row=0, column=7, rowspan=2, padx=10)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.ax.set_title("BER vs SNR (AWGN)")
        self.ax.set_xlabel("SNR (dB)")
        self.ax.set_ylabel("BER")
        self.ax.set_yscale("log")
        self.ax.grid(True, which="both", ls="--", lw=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

        # Table area
        table_frame = ttk.LabelFrame(root, text="QoS Metrics (Average)")
        table_frame.pack(fill="x", padx=10, pady=5)

        columns = ["Curve", "Modulation", "Avg BER", "Throughput (Mbps)", "Delay (ms)", "Spectral Eff (bits/s/Hz)"]
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings")

        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=130)

        self.table.pack(fill="x", padx=5, pady=5)

    # plot
    def add_curve(self):
        try:
            mod = self.mod_box.get()
            name = self.curve_name.get()
            snr_start = float(self.snr_start.get())
            snr_end = float(self.snr_end.get())
            points = int(self.points.get())

            snr_db = np.linspace(snr_start, snr_end, points)
            ber, metrics = simulate_ber_qos(mod, snr_db)

            self.ax.plot(snr_db, ber, label=f"{name} ({mod})")
            self.ax.legend()
            self.canvas.draw()

            self.curves_data.append({"Curve": name, "Modulation": mod, **metrics})
            self.table.insert("", "end", values=[name, mod, metrics["Avg BER"], metrics["Throughput (Mbps)"], metrics["Delay (ms)"], metrics["Spectral Eff (bits/s/Hz)"]])

        except Exception as e:
            messagebox.showerror("Error", f"Simulation error: {e}")

    def clear_curves(self):
        self.ax.clear()
        self.ax.set_title("BER vs SNR (AWGN)")
        self.ax.set_xlabel("SNR (dB)")
        self.ax.set_ylabel("BER")
        self.ax.set_yscale("log")
        self.ax.grid(True, which="both", ls="--", lw=0.5)
        self.canvas.draw()

        self.table.delete(*self.table.get_children())
        self.curves_data.clear()



if __name__ == "__main__":
    root = tk.Tk()
    app = ChannelSimApp(root)
    root.mainloop()
