import numpy as np
import cv2
import skimage as sk
from skimage.filters import gaussian
from skimage import transform
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
from skimage import transform
# =========================
# HELPERS
# =========================

def disk(radius, alias_blur=0.1):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    kernel = ((X**2 + Y**2) <= radius**2).astype(np.float32)
    kernel /= np.sum(kernel)
    return cv2.GaussianBlur(kernel, (3,3), sigmaX=alias_blur)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top+ch, top:top+ch], (zoom_factor, zoom_factor), order=1)
    trim = (img.shape[0] - h) // 2
    return img[trim:trim+h, trim:trim+h]

# =========================
# NOISE
# =========================

def gaussian_noise(x, severity=1):
    c = [0.08,0.12,0.18,0.26,0.38][severity-1]
    x = np.array(x)/255.0
    x = np.clip(x + np.random.normal(scale=c,size=x.shape),0,1)
    return (x*255).astype(np.float32)

def shot_noise(x, severity=1):
    c = [60,25,12,5,3][severity-1]
    x = np.array(x)/255.0
    x = np.clip(np.random.poisson(x*c)/c,0,1)
    return (x*255).astype(np.float32)

def impulse_noise(x, severity=1):
    c = [0.03,0.06,0.09,0.17,0.27][severity-1]
    x = sk.util.random_noise(np.array(x)/255.0, mode='s&p', amount=c)
    return (np.clip(x,0,1)*255).astype(np.float32)

# =========================
# BLUR
# =========================

def gaussian_blur(x, severity=1):
    c = [1,2,3,4,6][severity-1]
    x = gaussian(np.array(x)/255.0, sigma=c)
    return (np.clip(x,0,1)*255).astype(np.float32)

def defocus_blur(x, severity=1):
    c = [(3,0.1),(4,0.5),(6,0.5),(8,0.5),(10,0.5)][severity-1]
    x = np.array(x)/255.0
    kernel = disk(c[0], c[1])
    x = cv2.filter2D(x,-1,kernel)
    return (np.clip(x,0,1)*255).astype(np.float32)

def motion_blur(x, severity=1):
    c = [3,5,7,9,11][severity-1]
    x = np.array(x)/255.0
    kernel = np.zeros((c,c))
    if np.random.rand()>0.5:
        kernel[c//2,:]=1
    else:
        kernel[:,c//2]=1
    kernel/=c
    x = cv2.filter2D(x,-1,kernel)
    return (np.clip(x,0,1)*255).astype(np.float32)

def zoom_blur(x, severity=1):
    c = [
        np.arange(1,1.11,0.01),
        np.arange(1,1.16,0.01),
        np.arange(1,1.21,0.02),
        np.arange(1,1.26,0.02),
        np.arange(1,1.31,0.03)
    ][severity-1]

    x = np.array(x)/255.0
    out = np.zeros_like(x)
    for z in c:
        out += clipped_zoom(x,z)
    x = (x + out)/(len(c)+1)
    return (np.clip(x,0,1)*255).astype(np.float32)

# =========================
# WEATHER / VISUAL
# =========================

def fog(x, severity=1):
    c = [0.5,1.0,1.5,2.0,2.5][severity-1]
    x = np.array(x)/255.0
    noise = gaussian(np.random.uniform(size=x.shape), sigma=1)
    x = x + c*noise
    return (np.clip(x,0,1)*255).astype(np.float32)

def frost(x, severity=1):
    c = [0.2,0.3,0.4,0.5,0.6][severity-1]
    x = np.array(x)/255.0
    noise = gaussian(np.random.uniform(size=x.shape), sigma=1)
    x = np.clip(x + c*noise,0,1)
    return (x*255).astype(np.float32)

def snow(x, severity=1):
    c = [0.1,0.2,0.3,0.4,0.5][severity-1]
    x = np.array(x)/255.0
    snow = np.random.normal(loc=c, scale=0.3, size=x.shape)
    x = np.clip(x + snow,0,1)
    return (x*255).astype(np.float32)

# =========================
# COLOR / INTENSITY
# =========================

def brightness(x, severity=1):
    c = [0.1,0.2,0.3,0.4,0.5][severity-1]
    x = np.array(x)/255.0
    return (np.clip(x+c,0,1)*255).astype(np.float32)

def contrast(x, severity=1):
    c = [0.4,0.3,0.2,0.1,0.05][severity-1]
    x = np.array(x)/255.0
    m = np.mean(x)
    return (np.clip((x-m)*c+m,0,1)*255).astype(np.float32)

# =========================
# TRANSFORM
# =========================

from scipy.ndimage import map_coordinates

def elastic_transform(x, severity=1):
    c = [(34,4),(36,5),(38,6),(40,7),(42,8)][severity-1]

    x = np.array(x) / 255.0
    shape = x.shape

    dx = gaussian(np.random.uniform(-1,1,shape), sigma=c[1]) * c[0]
    dy = gaussian(np.random.uniform(-1,1,shape), sigma=c[1]) * c[0]

    x_coords, y_coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices = np.reshape(y_coords + dy, (-1, 1)), np.reshape(x_coords + dx, (-1, 1))

    distorted = map_coordinates(x, indices, order=1, mode='reflect').reshape(shape)

    return (np.clip(distorted, 0, 1) * 255).astype(np.float32)

# =========================
# DIGITAL
# =========================

def pixelate(x, severity=1):
    h,w = x.shape
    scale = [0.9,0.7,0.5,0.3,0.2][severity-1]
    x_small = cv2.resize(x,(int(w*scale),int(h*scale)))
    return cv2.resize(x_small,(w,h)).astype(np.float32)

from io import BytesIO
from PIL import Image

def jpeg_compression(x, severity=1):
    c = [80,60,40,25,10][severity-1]
    x = Image.fromarray(x.astype(np.uint8))
    buffer = BytesIO()
    x.save(buffer, format="JPEG", quality=c)
    x = Image.open(buffer)
    return np.array(x).astype(np.float32)
def speckle_noise(x, severity=1):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity-1]
    x = np.array(x) / 255.0
    x = np.clip(x + x * np.random.normal(scale=c, size=x.shape), 0, 1)
    return (x * 255).astype(np.float32)



def translate(x, severity=3):
    c = [1, 2, 3, 4, 5][severity-1]

    bit = np.random.choice([-1, 1], 2)
    dx = c * bit[0]
    dy = c * bit[1]

    aff = transform.AffineTransform(translation=[dx, dy])

    x = np.array(x) / 255.0
    x = transform.warp(x, inverse_map=aff)

    return (np.clip(x, 0, 1) * 255).astype(np.float32)

def rotate(x, severity=2):
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity-1]

    bit = np.random.choice([-1, 1], 1)[0]
    c *= bit

    aff = transform.AffineTransform(rotation=c)

    # keep image center fixed
    a1, a2 = aff.params[0, :2]
    b1, b2 = aff.params[1, :2]

    a3 = 13.5 * (1 - a1 - a2)
    b3 = 13.5 * (1 - b1 - b2)

    aff = transform.AffineTransform(rotation=c, translation=[a3, b3])

    x = np.array(x) / 255.0
    x = transform.warp(x, inverse_map=aff)

    return (np.clip(x, 0, 1) * 255).astype(np.float32)

def scale(x, severity=3):
    c = [(1/0.9, 1/0.9),
         (1/0.8, 1/0.8),
         (1/0.7, 1/0.7),
         (1/0.6, 1/0.6),
         (1/0.5, 1/0.5)][severity-1]

    aff = transform.AffineTransform(scale=c)

    # keep center fixed
    a1, a2 = aff.params[0, :2]
    b1, b2 = aff.params[1, :2]

    a3 = 13.5 * (1 - a1 - a2)
    b3 = 13.5 * (1 - b1 - b2)

    aff = transform.AffineTransform(scale=c, translation=[a3, b3])

    x = np.array(x) / 255.0
    x = transform.warp(x, inverse_map=aff)

    return (np.clip(x, 0, 1) * 255).astype(np.float32)

def spatter(x, severity=1):
    c = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1)
    ][severity-1]

    x = np.array(x, dtype=np.float32) / 255.0

    liquid_layer = np.random.normal(size=x.shape, loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])

    liquid_layer[liquid_layer < c[3]] = 0

    m = (liquid_layer > 0).astype(np.float32)
    m = gaussian(m, sigma=c[4])
    m[m < 0.8] = 0

    # mud-like effect
    color = 0.25 * m
    x = x * (1 - m) + color

    return (np.clip(x, 0, 1) * 255).astype(np.float32)