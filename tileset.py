import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import random
import pickle
import os

import functions as f
from arbitraryLattice import makeLatticeByAngles

class Tileset:
    default_padding = 100

    def __init__(self, path, cols, rows, tilew, tileh, scale, detection_method, ext='.tif'):
        self.path = path
        self.ext = ext
        self.rows = rows
        self.cols = cols
        self.tileh = tileh
        self.tilew = tilew
        self.scale = scale
        self.detection_method = detection_method
        self.blobs = np.array([])
        self.assigned_blobs = []
        self.lattice = None

    def getTile(self, col, row):
        """ Load from file and return a tile specified by row and column.
        If there is no tile at the specified position, returns an array of zeroes with the same size as a tile.

        :param col: the column of the tile to be returned
        :param row: the row of the tile to be returned
        :return: numpy array of the tile image
        """
        file_path = self.path + '/c_' + str(col) + '/tile_' + str(row) + self.ext
        try:
            tile = misc.imread(file_path)
        except FileNotFoundError:
            tile = np.zeros((self.tileh, self.tilew), dtype=np.uint8)

        return tile

    def getTileRegion(self, col_min, col_max, row_min, row_max):
        """Return a region of the image by concatenating a set of tiles"""
        r_width = self.tilew * (col_max - col_min + 1)
        r_height = self.tileh * (row_max - row_min + 1)
        region = np.zeros((r_height, r_width), dtype=np.uint8)

        for col in range(col_min, col_max+1):
            for row in range(row_min, row_max+1):

                tile = self.getTile(col, row)

                h_min = self.tileh * (row - row_min)
                h_max = self.tileh * (row - row_min + 1)
                w_min = self.tilew * (col - col_min)
                w_max = self.tilew * (col - col_min + 1)

                region[h_min:h_max, w_min:w_max] = tile

        return region

    def displayTileRegion(self, col_min, col_max, row_min, row_max, blob_color='red', lattice_color='cyan',
                          connector_color='yellow', figsize=(24, 12), path='', hide_axes=False, feature_scale=1):
        """Display a figure showing a region of the image, with blobs, lattice points and displacement vectors marked"""
        tiles = self.getTileRegion(col_min, col_max, row_min, row_max)

        x_min = self.tilew * col_min
        x_max = self.tilew * (col_max + 1) - 1
        y_min = self.tileh * row_min
        y_max = self.tileh * (row_max + 1) - 1

        blobs = self.getSubsetOfBlobs(x_min, x_max, y_min, y_max)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((x_min, x_max, y_max, y_min))

        plt.imshow(tiles, extent=[x_min, x_max, y_max, y_min], cmap='gray', interpolation='nearest')
        f.plotCircles(ax, blobs, fig, dict(color=blob_color, linewidth=2*feature_scale, fill=False))

        if self.lattice:
            lattice = self.getLattice()
            points = self.lattice.getLatticePoints(x_min, x_max, y_min, y_max)
            flip_points = np.fliplr(points)
            f.plotCircles(ax, flip_points, fig, dict(color=lattice_color, linewidth=10*feature_scale, fill=True))

            assigned_blobs = self.getAssignedBlobs()

            from matplotlib.collections import LineCollection
            from matplotlib.colors import colorConverter

            connectors = np.zeros((len(assigned_blobs), 2, 2), float)
            for i, a_blob in enumerate(assigned_blobs):
                if len(a_blob['point']) > 0:
                    bx = a_blob['blob'][1]
                    by = a_blob['blob'][0]
                    [px, py] = lattice.getCoordinates(*a_blob['point'])
                    connectors[i, :, :] = [[bx, by], [px, py]]

            colors = colorConverter.to_rgba(connector_color)
            line_segments = LineCollection(connectors, colors=colors, linewidths=2*feature_scale)
            ax.add_collection(line_segments)

        if hide_axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.tight_layout()

        if path == '':
            # plt.show()
            pass
        else:
            plt.savefig(path)
            print('Saved figure to', path)

    def getPaddedTile(self, col, row, padding=default_padding):
        """Return a tile with padding from adjacent tiles"""
        # TODO: Throw exception if padding is too large
        region = self.getTileRegion(col-1, col+1, row-1, row+1)

        h_crop = self.tileh - padding
        v_crop = self.tilew - padding

        padded = region[h_crop:-h_crop, v_crop:-v_crop]

        return padded

    def prepTiles(self, output_path, kernel_size, fill=True):
        """Do preprocessing on all tiles in tileset, and save the preprocessed tiles to output_path
        Preprocessing consists of filling in using reconstruction, and median filtering

        :param output_path: the path where the processed tiles will be saved
        :param kernel_size: width of the kernel used for median filtering
        :return: Tileset object containing the preprocessed tiles
        """
        from scipy.signal import medfilt

        for col in range(0, self.cols):
            col_path = output_path + '/c_' + str(col)
            if not os.path.exists(col_path):
                os.makedirs(col_path)

            for row in range(0, self.rows):
                tile = self.getPaddedTile(col, row)

                if fill:
                    tile = f.fillWires(tile)  # TODO: explain this

                tile = medfilt(tile, kernel_size)

                if fill:
                    tile = f.fillWires(tile)

                p = self.default_padding
                cropped_tile = tile[p:-p, p:-p]

                filename = col_path + '/tile_' + str(row) + self.ext
                misc.imsave(filename, cropped_tile)
                print('Saved tile ' + str(col) + ', ' + str(row))

        return Tileset(output_path, self.cols, self.rows, self.tilew, self.tileh, self.scale, self.detection_method, self.ext)

    def detectBlobs(self, col, row, globalize=False):
        """Run detection and return array of detected blobs for the specified tile

        :param col:
        :param row:
        :param globalize: if true, blobs will be returned with global coordinates
        :return:
        """
        padded_tile = self.getPaddedTile(col, row)

        blobs = self.detection_method(padded_tile)
        pad = self.default_padding

        outside = []
        for i, blob in enumerate(blobs):
            if min(blob[0:2]) < pad or blob[0] >= (self.tileh+pad) or blob[1] >= (self.tilew+pad):
                outside.append(i)

        blobs = np.delete(blobs, outside, 0)

        blobs[:, 0:2] -= pad
        if globalize:
            blobs[:, 1] += col * self.tilew
            blobs[:, 0] += row * self.tileh

        print('Blobs found:', blobs.shape[0])

        return blobs

    def plotBlobRegion(self, col_min, col_max, row_min, row_max, property='radius', hide_axes=False, colormap=''):
        """Show a figure plotting all detected blobs from the specified tile region without any background image"""
        x_min = self.tilew * col_min
        x_max = self.tilew * (col_max + 1) - 1
        y_min = self.tileh * row_min
        y_max = self.tileh * (row_max + 1) - 1

        label = ''
        if property == 'radius' or property == 'diameter':
            property = 'radius'
            label = 'diameter [nm]'
            if colormap == '': colormap = 'jet'
        elif property == 'displacement' or property == 'distance':
            property = 'distance'
            label = 'Displacement from lattice point [nm]'
            if colormap == '': colormap = 'viridis'
        elif property == 'angle':
            label = 'Angle of displacement from lattice point'
            if colormap == '': colormap = 'hsv'
        else:
            raise RuntimeError("Invalid property '" + str(property) + "'")

        def isInside(a_blob, x_min, x_max, y_min, y_max):
            inside = False
            blob_x = a_blob['blob'][1]
            blob_y = a_blob['blob'][0]
            if x_min <= blob_x <= x_max and y_min <= blob_y <= y_max:
                inside = True

            return inside

        assigned_blobs = self.getAssignedBlobs()
        assigned_blobs = [a_blob for a_blob in assigned_blobs if isInside(a_blob, x_min, x_max, y_min, y_max)]

        blobs = np.zeros((len(assigned_blobs), 4))
        for i, a_blob in enumerate(assigned_blobs):
            blobs[i, 0] = a_blob['blob'][0]
            blobs[i, 1] = a_blob['blob'][1]
            blobs[i, 2] = 45
            if property == 'radius':
                blobs[i, 3] = a_blob['blob'][2] * 2 * self.scale
            elif property == 'distance':
                blobs[i, 3] = a_blob['distance'] * self.scale
            elif property == 'angle':
                blobs[i, 3] = a_blob['angle']

        fig, ax = plt.subplots(figsize=(24, 12))
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((x_min, x_max, y_max, y_min))

        from matplotlib.collections import PatchCollection

        patches = []
        colors = []

        for circle in blobs:
            y, x, r, c = circle
            colors.append(c)
            patch = plt.Circle((x, y), r, linewidth=0, fill=True)
            patches.append(patch)

        p = PatchCollection(patches, match_original=True, cmap=colormap)
        p.set_array(np.array(colors))
        p.set_alpha(0.8)
        fig.colorbar(p, ax=ax, label=label)
        ax.add_collection(p)

        plt.tight_layout()
        if hide_axes:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.show()
        plt.close()

    def plotHistogram(self, property, bins=500, fontsize=20):
        if property == 'diameter':
            label = 'diameter [nm]'
            blobs = self.getBlobs()
            data = blobs[:, 2] * self.scale * 2
        elif property == 'distance':
            label = 'displacement from lattice point [nm]'
            assigned_blobs = self.getAssignedBlobs()
            data = [a_blob['distance'] * self.scale for a_blob in assigned_blobs]
        elif property == 'angle':
            label = 'angle'
            assigned_blobs = self.getAssignedBlobs()
            data = [a_blob['angle'] for a_blob in assigned_blobs]
        else:
            raise ValueError("'" + property + "' is not a valid property")

        fig, ax = plt.subplots(1, 1, figsize=(21.5, 10), subplot_kw={'adjustable': 'box-forced'})

        ax.set_title(property)
        ax.hist(data, bins=bins, histtype='stepfilled', edgecolor='none', color='#033A87')
        plt.xlabel(label, fontsize=fontsize)
        plt.ylabel('count', fontsize=fontsize)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        plt.tight_layout()
        plt.show()

    def plotRadialHistogram(self, property, bins=90, fontsize=20):
        if property == 'diameter':
            label = 'diameter [nm]'
            blobs = self.getBlobs()
            data = blobs[:, 2] * self.scale * 2
        elif property == 'distance':
            label = 'displacement from lattice point [nm]'
            assigned_blobs = self.getAssignedBlobs()
            data = [a_blob['distance'] * self.scale for a_blob in assigned_blobs]
        elif property == 'angle':
            label = 'angle'
            assigned_blobs = self.getAssignedBlobs()
            data = [a_blob['angle'] for a_blob in assigned_blobs]
        else:
            raise ValueError("'" + property + "' is not a valid property")

        ax = plt.subplot(111, projection='polar')

        ax.set_title(property)
        ax.hist(data, bins=bins, histtype='stepfilled', edgecolor='none', color='#033A87')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

        plt.show()

    @staticmethod
    def showMap(map):
        plt.imshow(map, cmap='viridis')
        plt.colorbar(label=r'Droplets per µm$^2$')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.show()
        plt.close()

    def getDensityMap(self, scale_factor, radius):
        from math import floor, ceil, pi

        sf = scale_factor
        r = ceil(radius/scale_factor)
        d = 2 * r  # diameter

        x_min = self.tilew * 0
        x_max = self.tilew * self.cols - 1
        x_len = x_max - x_min
        y_min = self.tileh * 0
        y_max = self.tileh * self.rows - 1
        y_len = y_max - y_min

        blobs = self.getBlobs()

        add_array = f.getCircleOfOnes(r)

        blob_points = [(floor(blob[0] / sf), floor(blob[1] / sf)) for blob in blobs]

        data = np.zeros((ceil(y_len / sf) + d, ceil(x_len / sf) + d))

        for point in blob_points:
            data[point[0]:point[0] + d, point[1]:point[1] + d] += add_array

        px_area = self.scale**2 / 10**6
        c_area = pi * radius**2 * px_area
        data = data / c_area

        return data

    def plotDensity(self, scale_factor, radius):
        map = self.getDensityMap(scale_factor, radius)
        self.showMap(map)

    def detectAllBlobs(self):
        """Run detection on all tiles, and save the result."""
        blobs = False
        for col in range(0, self.cols):
            for row in range(0, self.rows):
                found = self.detectBlobs(col, row, globalize=True)
                print('Detected blobs for tile ' + str(col) + ', ' + str(row))
                if blobs is False:
                    blobs = found
                else:
                    blobs = np.append(blobs, found, axis=0)

        self.blobs = blobs

        self.saveBlobs()

    def saveBlobs(self):
        """Store all currently detected blobs in a file."""
        full_path = self.path + '/blobs.p'
        pickle.dump(self.blobs, open(full_path, 'wb'))

    def deleteBlobs(self):
        full_path = self.path + '/blobs.p'
        try:
            os.remove(full_path)
        except FileNotFoundError:
            print('No file to delete')
        self.blobs = np.array([])

    def getBlobs(self):
        """Return all detected blobs for tileset. Load if possible, detect if necessary."""
        if self.blobs.shape[0] > 0:
            return self.blobs
        else:
            try:
                full_path = self.path + '/blobs.p'
                self.blobs = pickle.load(open(full_path, 'rb'))
                if self.blobs.shape[0] < 1:
                    print('Loaded blobs, but array was empty. Detecting blobs.')
                    self.detectAllBlobs()
            except FileNotFoundError:
                print('Blobs file not found. Detecting blobs.')
                self.detectAllBlobs()

        if self.blobs.shape[0] > 0:
            return self.blobs
        else:
            raise Exception('Not able to obtain blobs!')

    def getSubsetOfBlobs(self, x_min, x_max, y_min, y_max):
        """Return all detected blobs for specified coordinate region. Load if possible, detect if necessary."""
        blobs = self.getBlobs()

        outside = []
        for i, blob in enumerate(blobs):
            if blob[0] < y_min or blob[0] > y_max or blob[1] < x_min or blob[1] > x_max:
                outside.append(i)

        blobs = np.delete(blobs, outside, 0)

        return blobs

    def assignBlobs(self, blobs, lattice):
        """Assign a set of blobs to a lattice. Each blob is assigned to it's nearest lattice point.
        Return an array of dictionaries, each dictionary representing a blob, and containing the following:
        ['blob']: y, x, and r of the blob
        ['point']: lattice indices of the nearest lattice point
        ['distance']: absolute distance to the nearest lattice point
        ['angle']: angle of the displacement vector from blob to point
        """
        from scipy.spatial import KDTree

        assigned_blobs = [{'blob': blob, 'point': []} for blob in blobs]
        radius = lattice.getMinLatticeDist()/2

        x_min = min(blobs[:, 1]) - radius
        x_max = max(blobs[:, 1]) + radius
        y_min = min(blobs[:, 0]) - radius
        y_max = max(blobs[:, 0]) + radius

        points = lattice.getLatticePoints(x_min, x_max, y_min, y_max)
        tree = KDTree(points)

        for a_blob in assigned_blobs:
            y, x, r = a_blob['blob']
            distance, index = tree.query([x, y])
            point = tree.data[index]
            a_blob['point'] = lattice.getIndices(point[0], point[1])
            a_blob['distance'] = distance
            dis = np.array(point) - np.array([x, y])
            a_blob['angle'] = np.angle(dis[0] + 1j * dis[1])

        self.assigned_blobs = assigned_blobs
        self.saveAssignedBlobs()

        return assigned_blobs

    def saveAssignedBlobs(self):
        """Store all currently assigned blobs in a file."""
        full_path = self.path + '/assigned_blobs.p'
        pickle.dump(self.assigned_blobs, open(full_path, 'wb'))

    def deleteAssignedBlobs(self):
        full_path = self.path + '/assigned_blobs.p'
        try:
            os.remove(full_path)
        except FileNotFoundError:
            print('No file to delete')
        self.assigned_blobs = []

    def getAssignedBlobs(self):
        """Return assigned blobs for tileset. Load if possible, detect if necessary."""
        if len(self.assigned_blobs) > 0:
            return self.assigned_blobs
        else:
            try:
                full_path = self.path + '/assigned_blobs.p'
                self.assigned_blobs = pickle.load(open(full_path, 'rb'))
                if len(self.assigned_blobs) < 1:
                    print('Loaded assigned blobs, but array was empty. Assigning blobs.')
                    blobs = self.getBlobs()
                    lattice = self.getLattice()
                    self.assignBlobs(blobs, lattice)
            except FileNotFoundError:
                print('Assigned blobs file not found. Assigning.')
                blobs = self.getBlobs()
                lattice = self.getLattice()
                self.assignBlobs(blobs, lattice)

        if len(self.assigned_blobs) > 0:
            return self.assigned_blobs
        else:
            raise Exception('Not able to obtain assigned blobs!')

    @staticmethod
    def getAnglesFromInput(tile, blobs, offset):
        """Display an image with detected blobs plotted, and get user input to define angles for lattice vectors."""
        fig, ax = plt.subplots(figsize=(24, 12))
        ax.set_aspect('equal', adjustable='box-forced')
        plt.axis((0, 1023, 1023, 0))

        plt.imshow(tile, cmap='gray', interpolation='nearest')
        f.plotCircles(ax, blobs, fig, dict(color='#114400', linewidth=4, fill=False))
        plt.plot(offset[0], offset[1], 'o', color='red')
        plt.title("Please click two points")
        plt.tight_layout()

        # Get input
        points = plt.ginput(2)
        plt.close()

        # Calculate angles from input
        displacements = [np.array(point) - offset for point in points]
        angles = [np.angle(dis[0] + 1j*dis[1]) for dis in displacements]

        return angles

    @staticmethod
    def optimizeLattice(lattice, assigned_blobs, debug=False):
        """Use given lattice as an initial guess, and numerically optimize lattice to minimize
        the sum of square distances between each blob and it's nearest lattice point.
        """
        def getRSS(params, assigned_blobs):
            """Return the sum of square distances between each of the given blobs and it's nearest lattice point."""
            mag_a, ang_a, mag_b, ang_b, ox, oy = params
            lattice = makeLatticeByAngles(mag_a, ang_a, mag_b, ang_b, [ox, oy])

            sum = 0

            for blob_p in assigned_blobs:
                if blob_p['point'] != []:
                    blob_y, blob_x, r = blob_p['blob']
                    [point_x, point_y] = lattice.getCoordinates(*blob_p['point'])
                    square_dist = (point_x - blob_x) ** 2 + (point_y - blob_y) ** 2

                    sum += square_dist

            return sum

        from scipy.optimize import minimize
        params = np.array([lattice.len_a, lattice.ang_a, lattice.len_b, lattice.ang_b, lattice.offset[0], lattice.offset[1]])

        print('Blobs: ' + str(len(assigned_blobs)))

        res = minimize(getRSS, params, args=(assigned_blobs), method='Nelder-Mead')

        mag_a, ang_a, mag_b, ang_b, ox, oy = res['x']
        lattice = makeLatticeByAngles(mag_a, ang_a, mag_b, ang_b, [ox, oy])

        if debug:
            print(mag_a, ang_a, mag_b, ang_b, ox, oy)

        return lattice

    def makeLattice(self, max_blobs=500, final_blobs=4000, step=3, debug=False):
        """Run the whole process necessary to get a lattice defined for the tileset, and save it to file."""
        # Setup
        tw = self.tilew
        th = self.tileh
        bounds = (0, tw, 0, th)

        # The process starts with an initial guess based on the top left tile
        tile = self.getTile(0, 0)
        blobs = self.getSubsetOfBlobs(*bounds)
        # The top left blob is the offset of the initial guess
        first = self.findFirstBlob(blobs)
        offset = [first[1], first[0]]

        # Angles of the lattice vectors are given by manual input
        angles = self.getAnglesFromInput(tile, blobs, offset)
        if len(angles) < 2:
            raise RuntimeError("Insufficient input received.")
        # The magnitude of the lattice vectors is given by the typical nearest neighbor distance
        peak = self.getTypicalDistance(self.getSubsetOfBlobs(0, 4*tw, 0, 4*th))

        lattice = makeLatticeByAngles(peak, angles[0], peak, angles[1], offset)
        assigned_blobs = self.assignBlobs(blobs, lattice)

        lattice = self.optimizeLattice(lattice, assigned_blobs)
        print('Lattice optimized for first tile.')

        if debug:
            self.lattice = lattice  # needs to be set for displayTileRegion
            self.displayTileRegion(0, 0, 0, 0, blob_color='green', lattice_color='red')

        def optimizeWithBounds(self, lattice, bounds, max_blobs):
            blobs = self.getSubsetOfBlobs(*bounds)
            if blobs.shape[0] > max_blobs:
                blobs_list = list(blobs)
                blobs_list = [blobs_list[i] for i in random.sample(range(len(blobs_list)), max_blobs)]
                blobs = np.array(blobs_list)
            assigned_blobs = self.assignBlobs(blobs, lattice)
            lattice = self.optimizeLattice(lattice, assigned_blobs)

            return lattice

        # Gradually expand the area being optimized column by column
        for n in range(1, self.cols, step):
            bounds = (0, (n+1)*tw, 0, th)
            lattice = optimizeWithBounds(self, lattice, bounds, max_blobs)
            print('Lattice optimized for', n+1, 'of', self.cols, 'columns.')

        # Gradually expand the area being optimized row by row
        for n in range(1, self.rows, step):
            bounds = (0, self.cols*tw, 0, (n+1)*th)
            lattice = optimizeWithBounds(self, lattice, bounds, max_blobs)
            print('Lattice optimized for', n+1, 'of', self.rows, 'rows.')

        # Run one last optimization on the entire image, using a larger selection of blobs
        bounds = (0, self.cols * tw, 0, self.rows * th)
        lattice = optimizeWithBounds(self, lattice, bounds, final_blobs)
        print('Final optimization finished.')

        if debug:
            self.lattice = lattice  # needs to be set for displayTileRegion
            self.displayTileRegion(0, 0, 0, 0, blob_color='green', lattice_color='red')

        self.lattice = lattice
        self.saveLattice()
        self.deleteAssignedBlobs()

    def saveLattice(self):
        """Save the lattice stored in self.lattice to a file located in self.path"""
        full_path = self.path + '/lattice.p'
        pickle.dump(self.lattice, open(full_path, 'wb'))

    def deleteLattice(self):
        full_path = self.path + '/lattice.p'
        try:
            os.remove(full_path)
        except FileNotFoundError:
            print('No file to delete')
        self.lattice = None

    def getLattice(self):
        """Obtain a lattice by whatever means necessary. Try the following order:
        1: return self.lattice
        2: load lattice from file
        3: generate new lattice
        """
        if self.lattice != None:
            return self.lattice
        else:
            try:
                full_path = self.path + '/lattice.p'
                self.lattice = pickle.load(open(full_path, 'rb'))
                if self.lattice == None:
                    print('Loaded lattice, but array was empty.')
                    self.makeLattice()
            except FileNotFoundError:
                print('Lattice file not found!')
                self.makeLattice()

        return self.lattice

    @staticmethod
    def findFirstBlob(blobs):
        """Return the most top left blob in the given set of blobs"""
        vals = []
        for blob in blobs:
            vals.append(blob[0] + blob[1])

        i = np.argmin(vals)

        return blobs[i]

    @staticmethod
    def getTypicalDistance(blobs):

        def reject_outliers(data, m):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / mdev if mdev else 0.
            return data[s < m]

        from scipy.spatial import KDTree
        points = blobs[:, 0:2]

        tree = KDTree(points)
        results = [tree.query(points, 7)]

        distances = [result[1:7] for result in results[0][0]]
        distances = np.array(distances).flatten()
        distances = reject_outliers(distances, 5)

        return np.mean(distances)

    def getYield(self):
        blobs = self.getBlobs()
        lattice_points = self.getLattice().getLatticePoints(0, self.tilew * self.cols, 0, self.tileh * self.rows)

        the_yield = blobs.shape[0] / len(lattice_points)

        return the_yield
