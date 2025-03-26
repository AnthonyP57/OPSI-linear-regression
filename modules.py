import time
import matplotlib.pyplot as plt
import numpy as np

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r  took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result

    return timed

def xy_plot(x_test, y_test, pred, tlr_metrics, save_to, name=None):
  plt.plot((min(x_test), max(x_test)), (min(y_test), max(y_test)), linestyle='--', color='red', linewidth=1, label='True Values')
  plt.scatter(x_test, pred, s=10, linewidths=0.5, label=f'Predicted Values (R2={tlr_metrics[1]:.3f})')
  plt.xlabel('x')
  plt.ylabel('y')
  if name:
    plt.title(name)
  plt.legend()
  plt.savefig(save_to)
  plt.close()

def actual_vs_pred(y_test, pred, tlr_metrics, save_to, name=None):
  plt.plot((min(y_test), max(y_test)), (min(y_test), max(y_test)), linestyle='-', color='blue', linewidth=1, label='True Values')
  plt.scatter(y_test, pred, s=10, linewidths=0.5, label=f'Predicted Values (R2={tlr_metrics[1]:.3f})')
  plt.xlabel('Labels')
  plt.ylabel('Predictions')
  if name:
    plt.title(name)
  plt.legend()
  plt.savefig(save_to)
  plt.close()

def xyz_plot(x_test, y_test, pred, tlr_metrics, save_to, name=None):
  ax = plt.axes(projection="3d")

  z_line = (min(y_test.reshape(-1)), max(y_test.reshape(-1))) # y_test
  x_line = (min(x_test[:, 0].reshape(-1)), max(x_test[:, 0].reshape(-1)))
  y_line = (min(x_test[:, 1].reshape(-1)), max(x_test[:, 1].reshape(-1)))
  ax.plot3D(x_line, y_line, z_line, 'gray', label='True Values')

  z_points = pred.reshape(-1)
  x_points = x_test[:, 0].reshape(-1)
  y_points = x_test[:, 1].reshape(-1)
  ax.scatter3D(x_points, y_points, z_points, c=z_points, s=10, cmap='copper', label=f'Predicted Values (R2={tlr_metrics[1]:.3f})')
  ax.set_xlabel('x0')
  ax.set_ylabel('x1')
  ax.set_zlabel('y')
  if name:
    plt.title(name)
  plt.tight_layout()
  plt.legend()
  plt.savefig(save_to)
  plt.close()
