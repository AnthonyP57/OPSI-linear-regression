import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from RegressionVisualizer.data import ExponentialData, InverseData, LogData
from noise import Noise, UniformNoise
from regression import apply_regression
from plotting import create_plot

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Regression Visualizer")
        self.geometry("800x600")

        self.a = 1
        self.b = 2
        self.size = 16
        self.noise_type = tk.StringVar(value="None")
        self.noise_level = tk.DoubleVar(value=0.01)
        self.data_type = tk.StringVar(value="Exponential")
        self.plot_type = tk.StringVar(value="Original")
        self.canvas = None
        self.create_widgets()
        # Initial plot
        self.generate_plot()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side='left', fill='y', padx=10, pady=10)

        ttk.Label(control_frame, text="Data Type:").pack(anchor='w', pady=(0,5))
        for val in ["Exponential", "Inverse", "Logarithmic"]:
            ttk.Radiobutton(control_frame, text=val, variable=self.data_type, value=val, command=self.generate_plot).pack(anchor='w')

        ttk.Label(control_frame, text="Noise Type:").pack(anchor='w', pady=(10,5))
        for val in ["None", "Gaussian", "Uniform"]:
            ttk.Radiobutton(control_frame, text=val, variable=self.noise_type, value=val, command=self.generate_plot).pack(anchor='w')

        ttk.Label(control_frame, text="Noise Level:").pack(anchor='w', pady=(10,5))
        for lvl in [0.01, 0.05, 0.1]:
            ttk.Radiobutton(control_frame, text=str(lvl), variable=self.noise_level, value=lvl, command=self.generate_plot).pack(anchor='w')

        ttk.Label(control_frame, text="Plot Type:").pack(anchor='w', pady=(10,5))
        for val in ["Original", "Linearized", "Residuals", "All"]:
            ttk.Radiobutton(control_frame, text=val, variable=self.plot_type, value=val, command=self.generate_plot).pack(anchor='w')

        # Plot area
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side='right', fill='both', expand=True)

    def generate_plot(self):
        # Clear previous plot widget
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Generate data instance
        mapping = {'Exponential': ExponentialData, 'Inverse': InverseData, 'Logarithmic': LogData}
        DataClass = mapping[self.data_type.get()]
        data = DataClass(self.a, self.b, self.size)

        # Apply noise
        lvl = self.noise_level.get()
        if self.noise_type.get() == "Gaussian":
            data += Noise(0, lvl, self.size)
        elif self.noise_type.get() == "Uniform":
            data += UniformNoise(0, lvl, self.size)

        # Regression
        x_prime, y_prime = data.linearize()
        coef, intercept, r2, model = apply_regression(x_prime, y_prime)
        y_pred = model.predict(x_prime.reshape(-1, 1))

        # Build datasets
        datasets = []
        if self.plot_type.get() in ["Original", "All"]:
            datasets.append([{'x': data.x, 'y': data.y, 'label': 'Original', 'title': 'Original'}])
        if self.plot_type.get() in ["Linearized", "All"]:
            datasets.append([{'x_prime': x_prime, 'y_prime': y_prime, 'y_pred': y_pred, 'r2': r2, 'label': 'Linearized', 'title': f"Linearized ($R^2$={r2:.3f})"}])
        if self.plot_type.get() in ["Residuals", "All"]:
            residuals = y_prime - y_pred
            datasets.append([{'x': np.arange(len(residuals)), 'y': residuals, 'label': 'Residuals', 'title': 'Residuals'}])

        title = f"{self.data_type.get()} | {self.noise_type.get()} lvl {lvl} | {self.plot_type.get()}"
        fig = create_plot(datasets, title, linearized=self.plot_type.get() in ["Linearized","All"])

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        plt.close(fig)

if __name__ == "__main__":
    app = App()
    app.mainloop()