import numpy as np
import cv2
from PIL import Image
from skimage.filters import gaussian
import skimage as sk
from io import BytesIO
from scipy.ndimage import zoom as scizoom

# =========================
# Helpers
# =========================

def to_numpy(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    return x.astype(np.float32)

def to_float(x):
    return x / 255.0

def to_uint8(x):
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def clipped_zoom(img, zoom_factor):
    H, W = img.shape[:2]
    ch = int(np.ceil(H / zoom_factor))

    top = (H - ch) // 2
    img = scizoom(img[top:top+ch, top:top+ch], (zoom_factor, zoom_factor, 1), order=1)

    trim_top = (img.shape[0] - H) // 2
    return img[trim_top:trim_top+H, trim_top:trim_top+W]


# =========================
# 1. Noise (3)
# =========================

def gaussian_noise(x, severity=3):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity-1]
    x = to_float(to_numpy(x))
    return to_uint8(x + np.random.normal(0, c, x.shape))


def shot_noise(x, severity=3):
    c = [60, 25, 12, 5, 3][severity-1]
    x = to_float(to_numpy(x))
    return to_uint8(np.random.poisson(x * c) / c)


def impulse_noise(x, severity=3):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity-1]
    x = to_float(to_numpy(x))
    x = sk.util.random_noise(x, mode='s&p', amount=c)
    return to_uint8(x)


# =========================
# 2. Blur (3)
# =========================

def gaussian_blur(x, severity=3):
    c = [1, 2, 3, 4, 6][severity-1]
    x = to_float(to_numpy(x))
    return to_uint8(gaussian(x, sigma=c, channel_axis=-1))


def motion_blur(x, severity=3):
    c = [5, 10, 15, 20, 25][severity-1]
    x = to_numpy(x)

    kernel = np.zeros((c, c))
    kernel[c//2, :] = 1.0 / c

    return cv2.filter2D(x, -1, kernel)


def zoom_blur(x, severity=3):
    c = [np.arange(1,1.1,0.01),
         np.arange(1,1.15,0.01),
         np.arange(1,1.2,0.02),
         np.arange(1,1.25,0.02),
         np.arange(1,1.3,0.03)][severity-1]

    x = to_float(to_numpy(x))
    out = np.zeros_like(x)

    for zoom in c:
        out += clipped_zoom(x, zoom)

    return to_uint8((x + out) / (len(c)+1))


# =========================
# 3. Weather (3)
# =========================

def fog(x, severity=3):
    c = [0.2, 0.3, 0.4, 0.5, 0.6][severity-1]
    x = to_float(to_numpy(x))
    fog_layer = np.random.normal(loc=c, scale=0.1, size=x.shape)
    return to_uint8(x + fog_layer)


def snow(x, severity=3):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    x = to_float(to_numpy(x))
    snow_layer = np.random.normal(loc=c, scale=0.1, size=x.shape)
    return to_uint8(x + snow_layer)


def spatter(x, severity=3):
    c = [0.3, 0.4, 0.5, 0.6, 0.7][severity-1]
    x = to_float(to_numpy(x))
    liquid = np.random.normal(loc=c, scale=0.2, size=x.shape)
    mask = (liquid > 0.6).astype(np.float32)
    return to_uint8(x * (1-mask) + mask*0.5)


# =========================
# 4. Digital (3)
# =========================

def brightness(x, severity=3):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    x = to_float(to_numpy(x))
    return to_uint8(x + c)


def contrast(x, severity=3):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity-1]
    x = to_float(to_numpy(x))
    mean = np.mean(x, axis=(0,1), keepdims=True)
    return to_uint8((x - mean)*c + mean)


def jpeg_compression(x, severity=3):
    c = [25, 18, 15, 10, 7][severity-1]
    x = Image.fromarray(to_numpy(x).astype(np.uint8))

    buffer = BytesIO()
    x.save(buffer, "JPEG", quality=c)
    x = Image.open(buffer)

    return np.array(x).astype(np.float32)


# =========================
# 5. Geometry (3)
# =========================

def translate(x, severity=3):
    c = [1,2,3,4,5][severity-1]
    x = to_numpy(x)
    H, W = x.shape[:2]

    dx = np.random.randint(-c, c+1)
    dy = np.random.randint(-c, c+1)

    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(x, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def rotate(x, severity=3):
    c = [5,10,15,20,25][severity-1]
    x = to_numpy(x)
    H, W = x.shape[:2]

    angle = np.random.uniform(-c, c)
    M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)

    return cv2.warpAffine(x, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def scale(x, severity=3):
    c = [0.9,0.8,0.7,0.6,0.5][severity-1]
    x = to_numpy(x)
    H, W = x.shape[:2]

    new_h, new_w = int(H*c), int(W*c)
    resized = cv2.resize(x, (new_w, new_h))

    canvas = np.zeros_like(x)
    top = (H-new_h)//2
    left = (W-new_w)//2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas


# =========================
# FINAL LIST (15)
# =========================

CORRUPTIONS = [
    gaussian_noise,
    shot_noise,
    impulse_noise,

    gaussian_blur,
    motion_blur,
    zoom_blur,

    fog,
    snow,
    spatter,

    brightness,
    contrast,
    jpeg_compression,

    translate,
    rotate,
    scale,
]