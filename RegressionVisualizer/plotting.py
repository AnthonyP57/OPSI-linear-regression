import matplotlib.pyplot as plt
import numpy as np

def create_plot(datasets, title, linearized=False):
    """
    This function creates a line plot using matplotlib, sets the title and axis labels, and displays the plot.
    If out_path is specified, the plot is also saved to a file at the specified location.
    """
    nrows = len(datasets)
    ncols = len(datasets[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    if isinstance(axes, plt.Axes):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        if nrows == 1:
            axes = np.array([axes])
        else:
            axes = axes[:, np.newaxis]

    for i, row in enumerate(datasets):
        for j, data in enumerate(row):
            ax = axes[i][j]
            title_txt = data.get('title', 'Plot')
            ax.set_title(title_txt)
            if title_txt == 'Residuals':
                x = data['x']
                y = data['y']
                ax.bar(x, y)
                ax.set_xlabel('Index')
                ax.set_ylabel('Residual')
            else:
                x_key = 'x_prime' if linearized and 'x_prime' in data else 'x'
                y_key = 'y_prime' if linearized and 'y_prime' in data else 'y'
                x, y = data[x_key], data[y_key]
                ax.scatter(x, y, label=data.get('label', ''))
                if linearized and 'y_pred' in data:
                    ax.plot(x, data['y_pred'], linestyle='--', label=f"$R^2$={data['r2']:.3f}")
                ax.set_xlabel(x_key)
                ax.set_ylabel(y_key)
                ax.legend()
            ax.grid(True)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig