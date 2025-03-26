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
  plt.scatter(x_test, pred, s=10, linewidths=0.5, label=f'Predicted Values (R2={tlr_metrics[1]:.2f})')
  plt.xlabel('x')
  plt.ylabel('y')
  if name:
    plt.title(name)
  plt.legend()
  plt.savefig(save_to)
  plt.close()

def xyz_plot(x_test, y_test, pred, tlr_metrics, save_to, name=None):
    print(x_test.shape, y_test.shape, pred.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sorted_indices = np.argsort(x_test[:, 0])
    x_sorted = x_test[sorted_indices]
    y_sorted = y_test[sorted_indices]

    ax.plot(x_sorted[:, 0], x_sorted[:, 1], y_sorted, color='red', linewidth=2, label='True Values')

    ax.scatter(x_test[:, 0], x_test[:, 1], pred, color='blue', marker='x', label=f'Predicted Values (R2={tlr_metrics[1]:.2f})')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')

    if name:
        ax.set_title(name)

    ax.legend()
    plt.savefig(save_to)
    plt.close()