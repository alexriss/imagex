import matplotlib
import matplotlib.pylab as plt
import numpy as np
import skimage.feature
import time
import imagex


class Finder(object):
    """Class to find features in images. """
    
    def __init__(self, haystack, needle, ax_haystack=None, ax_needle=None, ax_needle_non_rotated=None, figsize=(12,6)):
        self.haystack = haystack       # haystack image
        self.needle = needle           # needle image
        self.plt_to_return = False     # generated plot, should be returned by the plot_coordinates function once
        
        if ax_haystack == None:
            self.plt_to_return = True
            self.plt_haystack_needle = plt.figure(figsize=figsize)  # figsize=(12,6)
            ax_haystack = plt.subplot(131)
            ax_needle = plt.subplot(132)
            ax_needle_non_rotated = plt.subplot(133)
        self.ax_haystack = ax_haystack  # figure axes for plotting the haystack
        self.ax_needle = ax_needle      # figure axes for plotting the needle
        self.ax_needle_non_rotated = ax_needle_non_rotated      # figure axes for plotting the needle (without rotation)
        imagex.ImageData.plot_data(None, haystack, name="Haystack", unit = "", pixel_units=True, axes=self.ax_haystack, extra_output=False)
    
    
    def plot_needle(self, coordinate, rotation, peak_value=None):
        """Plots needle from haystack image."""
        
        # clear previous stuff
        if len(self.ax_needle.images)>0:
            if self.ax_needle.images[-1].colorbar:
                self.ax_needle.images[-1].colorbar.remove()
        if self.ax_needle_non_rotated is not None:
           if len(self.ax_needle_non_rotated.images)>0:
                if self.ax_needle_non_rotated.images[-1].colorbar:
                    self.ax_needle_non_rotated.images[-1].colorbar.remove()

        c = coordinate
        deg = rotation
        # haystack image needs to be rotated accordingly
        haystack_rot = skimage.transform.rotate(self.haystack, deg, resize=True, cval=0)

        # transform coordinates as well
        center = np.array((self.haystack.shape[1], self.haystack.shape[0])) / 2. - 0.5
        tform1 = skimage.transform.SimilarityTransform(translation=center)
        tform2 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(deg))
        tform3 = skimage.transform.SimilarityTransform(translation=-center)
        tform = tform3 + tform2 + tform1
        c_rot = skimage.transform.matrix_transform(c, tform.params)[0]
        c_rot_start = np.array([int(c_rot[0]-self.needle.shape[0]/2), int(c_rot[1]-self.needle.shape[1]/2)])
        
        # the values matrix transformation need to be shifted, such that they correspond to the image starting at coordinates 0,0
        x_len, y_len = self.haystack.shape[1]-1, self.haystack.shape[0]-1
        c_rot_start = c_rot_start - 0.5 * (x_len - (x_len*np.abs(np.sin(np.deg2rad(deg))) + y_len*np.abs(np.cos(np.deg2rad(deg)))) )
        c_rot_start = np.round(c_rot_start).astype(np.int)
        
        needle_name = "found ("
        if peak_value > 0: needle_name += "value=%0.3f, " % peak_value
        needle_name += "%d degrees)" % deg
        
        imagex.ImageData.plot_data(None, haystack_rot[c_rot_start[0]:c_rot_start[0]+self.needle.shape[0], c_rot_start[1]:c_rot_start[1]+self.needle.shape[1]], name=needle_name, unit = "", pixel_units=True, axes=self.ax_needle, extra_output=False)
        
        if self.ax_needle_non_rotated is not None:
            # remove old rectangle
            for coll in self.ax_needle_non_rotated.collections[::-1]:
                coll.remove()
                
            # get needle image in original haystack (needs to be a bit bigger to encompass the rotated needle)
            c_center = np.array(c)
            x_len_needle, y_len_needle = self.needle.shape[1]-1, self.needle.shape[0]-1
            c_len = np.array([x_len_needle*np.abs(np.sin(np.deg2rad(deg))) + y_len_needle*np.abs(np.cos(np.deg2rad(deg))),
                              y_len_needle*np.abs(np.sin(np.deg2rad(deg))) + x_len_needle*np.abs(np.cos(np.deg2rad(deg)))])
            c_start = np.max([c_center-c_len/2,[0,0]], axis=0)
            c_end = np.min([c_center+c_len/2,[self.haystack.shape[0]-1,self.haystack.shape[1]-1]], axis=0)
            c_start = np.floor(c_start).astype(np.int)
            c_end = np.ceil(c_end).astype(np.int)

            imagex.ImageData.plot_data(None, self.haystack[c_start[0]:c_end[0], c_start[1]:c_end[1]], name="original rotation", unit = "", pixel_units=True, axes=self.ax_needle_non_rotated, extra_output=False)
            
            # plot rectangle
            t = matplotlib.transforms.Affine2D().rotate_deg_around(c[1]-c_start[1], c[0]-c_start[0], deg)
            rect = matplotlib.patches.Rectangle((c[1]-self.needle.shape[1]/2-0.5-c_start[1], c[0]-self.needle.shape[0]/2-0.5-c_start[0]), self.needle.shape[1], self.needle.shape[0], transform=t)
            rects_active = [rect]
            self.ax_needle_non_rotated.add_collection(matplotlib.collections.PatchCollection(rects_active, edgecolor='#ffff00', facecolor='none', linewidth=1, linestyle=":"))


    def plot_coordinates(self, coordinates, rotations, peak_values=[], i_needle=-1):
        """Plots the coordinates of the found needles in haystack.
        Parameter coordinates gives the x,y positions corresponding to the center of the match (as obtained by skimage.match_template(pad_input=True)).
        Parameter rotation gives the corresponding rotations of the matches.
        """
        # remove the previous patches from the plot
        for c in self.ax_haystack.collections[::-1]:
            c.remove()
        
        if i_needle < 0: i_needle = len(coordinates)-1

        rects = []
        rects_active = []
        if len(coordinates)>0:
            for i, (c,deg) in enumerate(zip(coordinates, rotations)):
                #ts = self.ax_haystack.transData
                t = matplotlib.transforms.Affine2D().rotate_deg_around(c[1], c[0], deg)
                rect = matplotlib.patches.Rectangle((c[1]-self.needle.shape[1]/2-0.5, c[0]-self.needle.shape[0]/2-0.5), self.needle.shape[1], self.needle.shape[0], transform=t)
                if i==i_needle:
                    rects_active.append(rect)
                else:
                    rects.append(rect)
            self.ax_haystack.add_collection(matplotlib.collections.PatchCollection(rects, edgecolor='#ff9000', facecolor='none', linewidth=1))
            self.ax_haystack.add_collection(matplotlib.collections.PatchCollection(rects_active, edgecolor='#ffff00', facecolor='none', linewidth=1, linestyle=":"))
            if self.ax_needle is not None:
                self.plot_needle(coordinates[i_needle], rotations[i_needle], peak_values[i_needle])
        
        #if self.plt_generated: return self.plt_haystack_needle
        if self.plt_to_return:
            self.plt_to_return = False
            return self.plt_haystack_needle
        
        
    def find_matches(self, threshold=0.8, angle_max=0, angle_stepsize=5, mirror=False):
        """Finds matches of needle in haystack.
        This function can also deal with rotations (parameters angle_max and angle_stepsize) and chirality (parameter mirror).
        """
        t = time.perf_counter()
        deg_range = np.arange(0,angle_max+angle_stepsize,angle_stepsize)
        results = np.empty((len(deg_range), self.haystack.shape[0], self.haystack.shape[1]))  # holding the result images
        results_angles = np.empty(len(deg_range))  # the corresponding angles of the result images
        for i,deg in enumerate(deg_range):
            r = skimage.feature.match_template(skimage.transform.rotate(self.haystack, deg, resize=True, cval=0), self.needle, pad_input=True)
            if len(np.where(r<-1)[0]):
                r[np.where(r<-1)] = -1  # had to include this thing, because skimage v0.13 gave values <-1 in some cases
                print("Warning: skimage_match_template gave values <-1. Replaced by -1.")
            # rotate back and cut accordingly
            offset_l = [int((r.shape[0] - self.haystack.shape[0])/2), int((r.shape[1] - self.haystack.shape[1])/2)]
            offset_r = [offset_l[0] + self.haystack.shape[0], offset_l[1] + self.haystack.shape[1]]
            
            results[i] = skimage.transform.rotate(r, -deg, resize=False, cval=0)[offset_l[0]:offset_r[0], offset_l[1]:offset_r[1]]
            results_angles[i] = deg
        # get highest value of all rotation of the matching-result images
        result = np.max(np.array(results), axis=0)
        result_i = np.argmax(np.array(results), axis=0)
        coordinates = skimage.feature.peak_local_max(result, min_distance=np.min(self.needle.shape), threshold_rel=threshold, indices=True)
        peak_values = result[coordinates[:,0],coordinates[:,1]]
        rotations = results_angles[result_i[coordinates[:,0],coordinates[:,1]]]
        
        # sort according to highest match score
        i_sorted = np.argsort(peak_values)
        peak_values = peak_values[i_sorted[::-1]]
        coordinates = coordinates[i_sorted[::-1]]
        rotations = rotations[i_sorted[::-1]]
        print("Found: %d   (%d ms)" % (len(coordinates), 1e3*(time.perf_counter()-t)))
        
        return coordinates, rotations, peak_values


