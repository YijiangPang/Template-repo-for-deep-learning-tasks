import os
import numpy as np
import matplotlib.pyplot as plt


class Surface_3D_plot():
    def __init__(self, x_range, y_range, resolution):
        x_range = x_range if isinstance(x_range, list) else [-x_range, x_range]
        y_range = y_range if isinstance(y_range, list) else [-y_range, y_range]
        x, y = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0])/resolution)), np.linspace(y_range[0], y_range[1], int((y_range[1] - y_range[0])/resolution))
        self.x, self.y = np.meshgrid(x, y)

    def plot(self, path, name, zs, traj = None):
        zs = zs.reshape(self.x.shape)
        fig = plt.figure(figsize=(16,8))#
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        z_b, z_t = zs.min(), zs.max()

        ax1.set_xlabel('x0')
        ax1.set_ylabel('x1')
        ax1.set_zlabel('loss',rotation="vertical")
        # ax1.view_init(50, 290)
        ax1.plot_surface(self.x, self.y, zs, alpha=0.7, cmap='viridis')
        ax1.set_title(name)  
        ax1.axes.set_zlim3d(bottom=z_b, top=z_t)
        if traj is not None:
          for i, n in enumerate(traj):
            x_traj, y_traj, loss_traj = traj[n]
            ax1.plot3D(x_traj, y_traj, loss_traj, lw=2, label = n, marker = "o")
          ax1.legend()

        ax2.set_xlabel('x0')
        ax2.set_ylabel('x1')
        ax2.contour(self.x, self.y, zs, 20, cmap="viridis", linestyles="solid")
        if traj is not None:
          for i, n in enumerate(traj):
            x_traj, y_traj, loss_traj = traj[n]
            ax2.plot(x_traj, y_traj, lw=2, label = n, marker = "o")  #c = 'black'
            if i == 0:
              margin = 0.05
              ax2.text(x_traj[0] + margin, y_traj[0] + margin, "start", color='black')
              ax2.text(x_traj[-1] + margin, y_traj[-1] + margin, "end", color='black')
          ax2.legend()
        plt.show() 
        # plt.savefig("%s.pdf"%(os.path.join(path, "%s.png"%(name))), bbox_inches="tight")
        plt.savefig("%s.png"%(os.path.join(path, "%s"%(name))), bbox_inches="tight")  #
        plt.close()

if __name__ == "__main__":
   
  def loss(a,b):
    return ((a**2 - 1)**2)*(1+b**2) + 0.2*(b**2)

  def grad(a,b):
    grad_a = 4*(a**2 - 1)*a*(1 + b**2)
    grad_b = 2*((a**2 - 1)**2)*b + 0.4*b
    return grad_a, grad_b

  eta = 0.05
  a_ = 0.001
  b_ = 0.5
  x_traj = a_
  y_traj = b_
  loss_traj= loss(a_,b_)
  for i in range(20):
      grad_a, grad_b  = grad(a_,b_)
      a_ = a_ - eta * grad_a
      b_ = b_ - eta * grad_b
      loss_= loss(a_,b_)
      x_traj =np.append(x_traj,a_) 
      y_traj =np.append(y_traj,b_) 
      loss_traj =np.append(loss_traj,loss_) 
  traj0 = [x_traj, y_traj, loss_traj]

  eta = 0.05
  a_ = 0.001
  b_ = 0.5
  x_traj = a_
  y_traj = b_
  loss_traj= loss(a_,b_)
  for i in range(20):
      grad_a, grad_b  = grad(a_,b_)
      grad_norm = np.sqrt(grad_a**2 + grad_b**2)
      grad_a, grad_b = grad_a/grad_norm, grad_b/grad_norm
      a_ = a_ - eta * grad_a
      b_ = b_ - eta * grad_b
      loss_= loss(a_,b_)
      x_traj =np.append(x_traj,a_) 
      y_traj =np.append(y_traj,b_) 
      loss_traj =np.append(loss_traj,loss_) 
  traj1 = [x_traj, y_traj, loss_traj]

  #plot
  loss_plot = Surface_plot(x_range = [-1.5, 1.5], y_range = [-1.5, 1.5], resolution = 0.05)
  zs = np.array([loss(a,b) for a,b in zip(np.ravel(loss_plot.x), np.ravel(loss_plot.y))])
  traj = {"traj0": traj0, "traj1":traj1}
  loss_plot.plot("~/", "plot" ,zs, traj)
