import matplotlib.pyplot as plt

def visualize_loss(loss, label, title):
  epochs = range(1, len(loss) + 1)

  fig, ax = plt.subplots(1, figsize=(12, 5))
  ax.plot(epochs, loss, label=label)
  ax.set_xlabel("Epoch", fontsize=12)
  ax.set_ylabel(label, fontsize=12)
  ax.set_title(title, fontsize=16)

def visualize_losses(losses, labels, title):
  epochs = range(1, len(losses[0]) + 1)

  fig, ax = plt.subplots(1, figsize=(12, 5))

  for idx, loss in enumerate(losses):
    ax.plot(epochs, loss, label=labels[idx])

  ax.set_xlabel("Epoch", fontsize=12)
  ax.set_ylabel("Loss", fontsize=12)
  ax.set_title(title, fontsize=16)
  ax.legend(loc="best", fontsize=18)

def plot_line(x, y, ax, label=""):
  ax.plot(x, y, label=label)
