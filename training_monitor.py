from images import plot_images, save_image

class TrainingMonitor:
  def __init__(self, plot, save_it, save_it_dir, p0, a0):
    self.plot = plot
    self.save_it = save_it
    self.save_it_dir = save_it_dir
    self.p0 = p0
    self.a0 = a0

  def monitor_iteration(self, i, x, l, cl, sl, tvl):
    print(f"Iteration: {i}|loss {l}|{cl}|{sl}|{tvl}")
    if (i % 10 == 0):
      if self.plot: self._plot_images(self.p0, x, self.a0)
      if self.save_it: self._save_image(x, i)

  def _save_image(self, img, i):
    path = self._image_path(i)
    save_image(img, path)

  def _image_path(self, i):
    return f"{self.save_it_dir}/it{i}.jpg"

  def _plot_images(self, p, x, a):
    plot_images(content_image=p, style_image=a, mixed_image=x)
