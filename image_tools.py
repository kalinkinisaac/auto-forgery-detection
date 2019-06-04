import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def to_jpeg(image: Image, quality=100):
    # RGB only
    out = BytesIO()
    image.save(out, 'JPEG', subsampling=0, quality=quality)
    out.seek(0)
    return Image.open(out)


def self_tampered(image: Image, box=(0, 0, 64, 64), q0=60, q1=80):
    tamper = image.copy()#Image.blend(image.copy().convert('RGBA'), Image.new('RGBA', size=image.size, color=(218, 112, 214)), alpha=0.5).convert('RGB')
    tamper = to_jpeg(tamper, q0)
    tamper = tamper.crop(box)
    tampered = image.copy()
    tampered.paste(tamper, box)

    return to_jpeg(tampered, quality=q1)


def save_self_tampered(fp, image: Image, box=(0, 0, 64, 64), q0=60, q1=80):
    tamper = image.copy()
    tamper = to_jpeg(tamper, q0)
    tamper = tamper.crop(box)
    tampered = image.copy()
    tampered.paste(tamper, box)
    tampered.save(fp, 'JPEG', quality=q1)


def color_box(image: Image, box=(0, 0, 64, 64), color=(255,0,0)):
    tamper = image.copy().convert('RGBA')
    tamper = Image.blend(tamper, Image.new('RGBA', size=tamper.size, color=color), alpha=0.5)
    tamper = tamper.crop(box)
    tampered = image.copy()
    tampered.paste(tamper, box)
    return tampered
