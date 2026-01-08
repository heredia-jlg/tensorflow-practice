import numpy as np
import matplotlib.pyplot as plt

red_rbg = [255, 0, 0]
green_rbg = [0, 255, 0]
blue_rbg = [0, 0, 255]
white_rbg = [255, 255, 255]

def visualize_image(image: np.uint8):
    plt.imshow(image)
    plt.show()

def create_image(size: int, color: str) -> np.uint8:
    """ Create a simple image with a gradient """
    if color == 'red':
        return np.full((size, size, 3), red_rbg, dtype=np.uint8)
    elif color == 'green':
        return np.full((size, size, 3), green_rbg, dtype=np.uint8)
    elif color == 'blue':
        return np.full((size, size, 3), blue_rbg, dtype=np.uint8)
    else:
        return np.full((size, size, 3), white_rbg, dtype=np.uint8)