import numpy as np
import matplotlib.pyplot as plt

def fillWires(image):
    from skimage.morphology import reconstruction

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image

    dilated = reconstruction(seed, mask, method='erosion')

    return dilated

def plotCircles(axes, circles, fig, kwargs):
    from matplotlib.collections import PatchCollection

    patches = []

    for circle in circles:
        if len(circle) == 3:
            y, x, r = circle
        elif len(circle) == 2:
            y, x = circle
            r = 1
        else:
            raise RuntimeError('Wrong number of elements to define circle: ' + str(len(circle)))
        patch = plt.Circle((x, y), r, **kwargs)
        patches.append(patch)

    p = PatchCollection(patches, match_original=True)
    axes.add_collection(p)

def getCircleOfOnes(radius):
    diameter = 2 * radius
    ones = np.ones((diameter, diameter))
    y, x = np.ogrid[-radius: radius, -radius: radius]
    mask = x ** 2 + y ** 2 <= radius ** 2
    circle = ones * mask

    return circle
