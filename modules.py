import time
import matplotlib.pyplot as plt

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
      plt.title('Torch Linear Regression Backpropagation')
    plt.legend()
    plt.savefig(save_to)
    plt.close()
