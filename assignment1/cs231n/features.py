#coding:utf8
import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
  """
  Given pixel data for images and several feature functions that can operate on
  single images, apply all feature functions to all images, concatenating the
  feature vectors for each image and storing the features for all images in
  a single matrix.

  Inputs:
  - imgs: N x H X W X C array of pixel data for N images.
  - feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
  - verbose: Boolean; if true, print progress.

  Returns:
  An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
  of all features for a single image.
  """
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])

  # Use the first image to determine feature dimensions
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)

  # Now that we know the dimensions of the features, we can allocate a single
  # big array to store all features as columns.
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T

  # Extract features for the rest of the images.
  for i in xrange(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx
    if verbose and i % 1000 == 0:
      print 'Done extracting features for %d / %d images' % (i, num_images)

  return imgs_features


def rgb2gray(rgb):
  """Convert RGB image to grayscale

    Parameters:
      rgb : RGB image

    Returns:
      gray : grayscale image
  
  """
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def hog_feature(im):
  """Compute Histogram of Gradient (HOG) feature for an image
  
       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
     
     Reference:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005
     
    Parameters:
      im : an input grayscale or rgb image
      
    Returns:
      feat: Histogram of Gradient (HOG) feature
    
  """
  
  # convert rgb to grayscale if needed
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)
  sx, sy = image.shape # 图形大小  32x32像素
  orientations = 9 # 梯度直方图bin的数量
  cx, cy = (8, 8) # 每个cell包含的像素

  gx = np.zeros(image.shape) # 32x32
  gy = np.zeros(image.shape) # 32x32
  gx[:, :-1] = np.diff(image, n=1, axis=1) # 计算x轴方向梯度  gx[:,:-1]表示不包括gx最后一列，即gx[:,-1]不变，仍然全是0
  gy[:-1, :] = np.diff(image, n=1, axis=0) # 计算y轴方向梯度
  
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) # 梯度大小  32x32
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # 梯度方向  32x32

  n_cellsx = int(np.floor(sx / cx))  # x轴方向cell的数量 4
  n_cellsy = int(np.floor(sy / cy))  # y轴方向cell的数量 4

  # 计算完整图形的方向
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations)) # 4x4x9
  for i in range(orientations):
    # np.where相当于C语言中的三目运算符  grad_ori <（180/orientations*(i + 1)) ? grad_ori : 0
    #当i=0时，意味着grad_ori中<0 或>20的元素 对应于temp_ori的是0。意味着只保留 0<=grad_ori<20的元素。
    #直观上，就是0<=grad_ori<20的像素点有投票权。
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)  # 32x32
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    
    # 选择那些方向的大小
    cond2 = temp_ori > 0
    # temp_ori > 0 ? grad_mag : 0
    # 用梯度的大小作为投票的权重
    temp_mag = np.where(cond2, grad_mag, 0)
    #均匀滤波器
    local_mean = uniform_filter(temp_mag, size=(cx, cy))
    orientation_histogram[:,:,i] = local_mean[cx/2::cx, cy/2::cy].T
  return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  """
  Compute color histogram for an image using hue.

  Inputs:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)

  # return histogram
  return imhist


pass
