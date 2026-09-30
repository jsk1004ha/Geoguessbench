"""Actual equirectangular-to-perspective projection, not a moving crop simulation."""
import io
import math

import numpy as np
from PIL import Image, ImageOps


def perspective(image: Image.Image, yaw: float, pitch: float, fov: float,
                width: int, height: int) -> Image.Image:
    scale = math.tan(math.radians(fov) / 2)
    x, y = np.meshgrid((2 * (np.arange(width) + 0.5) / width - 1) * scale,
                       (1 - 2 * (np.arange(height) + 0.5) / height) * scale * height / width)
    z = np.ones_like(x)
    p, a = math.radians(pitch), math.radians(yaw)
    yy, zz = y * math.cos(p) + z * math.sin(p), z * math.cos(p) - y * math.sin(p)
    xx, zz = x * math.cos(a) + zz * math.sin(a), zz * math.cos(a) - x * math.sin(a)
    longitude = np.arctan2(xx, zz)
    latitude = np.arctan2(yy, np.sqrt(xx * xx + zz * zz))
    raw = np.asarray(image.convert("RGB"))
    h, w = raw.shape[:2]
    u = ((longitude / (2 * math.pi) + 0.5) * w - 0.5) % w
    v = np.clip((0.5 - latitude / math.pi) * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = (x0 + 1) % w, np.minimum(y0 + 1, h - 1)
    dx, dy = (u - x0)[..., None], (v - y0)[..., None]
    result = ((1-dx) * (1-dy) * raw[y0, x0] + dx * (1-dy) * raw[y0, x1]
              + (1-dx) * dy * raw[y1, x0] + dx * dy * raw[y1, x1])
    return Image.fromarray(np.clip(result, 0, 255).astype("uint8"))


def render_view(dataset, node, protocol, heading, pitch, fov) -> bytes:
    with Image.open(dataset.path_for(node)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if protocol.lower_band_mask:
            # Ablation only: this removes a fixed ground band, not all camera/location meta.
            top = round(image.height * (1 - protocol.lower_band_mask))
            image.paste((96, 96, 96), (0, top, image.width, image.height))
        if node.panorama:
            image = perspective(image, heading - node.heading, pitch, fov,
                                protocol.width, protocol.height)
        else:
            image = ImageOps.pad(image, (protocol.width, protocol.height), color=(24, 24, 24))
        # Recreate pixels: remove GPS EXIF, XMP, comments, filenames, ICC and source URLs.
        clean = Image.frombytes("RGB", image.size, image.tobytes())
        output = io.BytesIO()
        clean.save(output, "JPEG", quality=90, optimize=False)
        return output.getvalue()
