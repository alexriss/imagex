"""
imagex

SPM image import and analysis

Class to read Nanonis sxm files (SPM images). Currently not much analysis (yet).


2016, Alex Riss, GPL

parts of the load-file function are based on code from the Nanonis manual, written by Felix Wählisch (Beer-ware license).

"""



from __future__ import print_function
from __future__ import division
import sys
import re
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import skimage.measure
import ipywidgets
import datetime


import seaborn as sns
sns.set_style("dark")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.5})
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

PYTHON_VERSION = sys.version_info.major

cdict = {'red':   ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0),)}
greys_linear = matplotlib.colors.LinearSegmentedColormap('greys_linear', cdict)  # always have trouble with the brightness values
#plt.register_cmap(cmap=greys_linear)


class ImageData(object):
    """Class to read grid a Nanonis sxm image and analyze it. """

    def __init__(self):
        self.filename = ""         # filename
        self.header = {}           # dictionary containing all the header data
        self.channels = []         # list holding a dictionary with data_header (names, units and the swept channels) and data (np.array of the actual data)
        self.channel_names = []    # list holding the channel names
        
        self.scansize = (0,0)      # scan size
        self.scansize_unit = ""    # unit for scansize (e.g. nm)
        self.pixelsize = (0,0)     # scan size in pixels
        self.scan_direction = ""   # scan direction ("up" or "down")
        
        self.start_time = datetime.date.today() # start time of scan
        self.acquisition_time = 0   # aquisition time in seconds


    def load_image(self,fname,long_output=False):
        """loads nanonis SPM data from an sxm file."""
        ext = fname.rsplit(".",1)[-1]
        if  ext == "sxm":
            self.__init__()
            self._load_image_header(fname,long_output)
            self._load_image_body(fname,long_output)
        else:
            print("Error: Unknown file type \"%s\"." % ext)
            return False
        self.filename = fname


    def get_channel(self, channel_name):
        """gets the channel dictionary for channel_name"""
        if channel_name not in self.channel_names:
            if channel_name[-4:] not in ("_fwd", "_bwd"):
                channel_name += '_fwd'
                print('Using channel %s' % channel_name)
            if channel_name not in self.channel_names:
                print('Error: Channel %s does not exist in %s.' % (channel_name, self.filename))
                print('    Available channels are: %s.' % ", ".join(self.channel_names))
                return False

        i_channel = self.channel_names.index(channel_name)
        return self.channels[i_channel]


      
    def plot_image(self, channel_name, cmap='', interpolation="bicubic", background="none", no_labels=False, output_filename="", output_dpi=100, empty_image=False):
        '''return plot of images of channel. You can specify a colormap (cmap), the interpolation type, as well as background subtraction ("offset" or "none").
        If no_labels is set to True, then the bare body is output. If output_filename is given, then the plot is saved under this filename with resolution output_dpi.
        If empty_image is True, then it will just give out an empty image.'''

        channel = self.get_channel(channel_name)
        
        # z = np.fliplr(np.rot90(self.channels[i_channel]['data'], k=1))
        z = channel['data']
        unit = channel['data_header']['unit']
        name = channel['data_header']['name']
        return self.plot_data(z, name=name, unit=unit, cmap=cmap, interpolation=interpolation, background=background, no_labels=no_labels, output_filename=output_filename, output_dpi=output_dpi, alpha=alpha)
        
            
    def plot_data(self, data, name="", unit="", cmap='', interpolation="bicubic", background="none", no_labels=False, output_filename="", output_dpi=100, alpha=1):
        """returns plot of the data. You can specify a colormap (cmap), the interpolation type, as well as background subtraction ("offset" or "none").
        If no_labels is set to True, then the bare body is output. If output_filename is given, then the plot is saved under this filename with resolution output_dpi.
        name and unit are for the axis labels.
        If empty_image is True, then it will just give out a fully transparent empty image."""
    
        x_len, y_len = self.scansize[0], self.scansize[1]
        z = data
        # if unit == 'm':
            # z = z*1.0e9
            # unit = 'n'+unit
        if background == "offset":
            z_orig = z
            z = z - np.amin(z)

        if cmap=='': cmap=greys_linear

        fig = plt.figure()
        if no_labels:
            ax = fig.add_axes([0,0,1,1])
        else:
            ax = plt.subplot(111)
        img = ax.imshow(z, cmap=cmap, aspect='equal', interpolation=interpolation, origin='lower', picker=True, alpha=alpha)
        plt.setp(img, extent=(0,x_len,0,y_len))
        
        ax.grid(False)
        if no_labels:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_xlabel("X [nm]")
            ax.set_ylabel("Y [nm]")

            fig_title = name + ' ['+unit+']'
            if background != "none":
                fig_title += "; background correction: %s" % background
            ax.set_title(fig_title)
           
            divider =  mpl_toolkits.axes_grid1.make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = plt.colorbar(img, cax=cax)
            cbar.ax.tick_params(labelsize=8) 
            #cbar = plt.colorbar(img, shrink=0.8) # fraction=0.046, pad=0.04, shrink=0.8
            #cbar.set_label(name+' ['+unit+']')
        
        print("%s: min=%s%s, max=%s%s" % (name, np.amin(z), unit, np.amax(z), unit))
        if background != "none":
            print("%s (without background correction): min=%s%s, max=%s%s" % (name, np.amin(z_orig), unit, np.amax(z_orig), unit))
        if output_filename != "":
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=output_dpi)
            print("Saved: %s" % output_filename)

        return fig
        
        
    def line_profile(self, data, points=[], linewidth=1, outside_range='constant'):
        """gives out a line profile along the points array (x,y coordinates in nm units). Width specifies an optional thickness of the line"""
        
        if len(points)<2:
            print("Error: at least two points are needed for the line profile.")
            return False
        
        points_px = []
        for p in points:
            points_px.append(self.nm_to_pixels(p))
            
        lines = []
        x_pos = 0
        
        for i in range(len(points)-1):
            p1 = (points_px[i][1], points_px[i][0])       # for the profile_line function we need row first (i.e. the y coordinate), then the column (i.e. the x coordinate)
            p2 = (points_px[i+1][1], points_px[i+1][0])
            y = skimage.measure.profile_line(data, p1, p2, linewidth=linewidth, mode=outside_range)
            line_length = get_distance(points[i], points[i+1])
            x = np.linspace(x_pos, line_length, len(y))
            x_pos += line_length
            lines.append(np.array([x,y]))
        
        if len(lines) == 1:
            return lines[0]
        else:
            return lines


    def _load_image_header(self,fname,long_output):
        """load header data from a .sxm file"""

        print('Reading header of %s' % fname)
        if PYTHON_VERSION>=3:
            f = open(fname, encoding='utf-8', errors='ignore')
        else:
            f = open(fname)
        header_ended = False
        caption = re.compile(':*:')
        key = ''
        contents = ''
        while not header_ended:
            line = f.readline()
            if line == ":SCANIT_END:\n":  # end of header
                header_ended = True
                self.header[key] = contents
            else:
                if caption.match(line) != None:
                    if key != '':
                        self.header[key] = contents.strip()
                    key = string_simplify(line[1:-2])  # set new name
                    contents = ''  # reset contents
                else:  # if not caption, it is content
                    contents+=(line)
                # [todo: add some parsing here]
        f.close()
        
        x_len, y_len = self.header['scan_range'].split()
        self.scansize = (float(x_len)*1e9, float(y_len)*1e9)  # convert to nm
        self.scansize_unit = 'nm'
        xPixels, yPixels = self.header['scan_pixels'].split()
        self.pixelsize = (int(xPixels), int(yPixels))
        self.scan_direction = self.header['scan_dir']
        
        self.start_time = datetime.datetime.strptime(self.header['rec_date'] + " " + self.header['rec_time'], '%d.%m.%Y %H:%M:%S')
        self.acquisition_time = float(self.header['acq_time'])


    def _load_image_body(self,fname,long_output):
        """load body data from a .sxm file"""

        # extract channels to be read in
        print('Reading body of %s' % fname)
        xPixels, yPixels = self.pixelsize
        data_info = self.header['data_info']
        lines = data_info.split('\n')
        lines.pop(0)  # headers: Channel Name Unit Direction Calibration Offset
        names = []
        units = []
        for line in lines:
            entries = line.split()
            if len(entries) > 1:
                names.append(entries[1])
                units.append(entries[2])
                if entries[3] != 'both':
                    print("Error: Only one direction recorded. This is not implemented yet. (%s)" % entries)
                    return False

        f = open(fname, 'rb') #read binary
        read_all = f.read()
        offset = read_all.find(b'\x1A\x04')
        f.seek(offset+2)  # data start 2 bytes afterwards
        fmt = '>f' # float
        ItemSize = struct.calcsize(fmt)
        for i in range(len(names)*2): # fwd + bwd
            if i%2 == 0:
                direction = '_fwd'
            else:
                direction = '_bwd'
            bindata = f.read(ItemSize*xPixels*yPixels)
            data = np.zeros(xPixels*yPixels)
            for j in range(xPixels*yPixels):
                data[j] = struct.unpack(fmt, bindata[j*ItemSize: j*ItemSize+ItemSize])[0]
            data = data.reshape(yPixels, xPixels)
            # data = np.rot90(data)
            if direction == '_bwd':
                data = data[::-1]
            if self.scan_direction == "down":
                data = np.flipud(data)
            channel = {'data_header': {'name': names[int(i/2)]+direction, 'unit': units[int(i/2)]}, 'data': data}
            if long_output:
                print("  read: %s in %s, shape: %s" % channel['data_header'].name, channel['data_header'].unit, channel['data'].shape)
            self.channels.append(channel)
            self.channel_names.append(channel['data_header']['name'])
        f.close()

        
    def nm_to_pixels(self, p):
        """converts x,y coordinates from nm to pixel coordinates"""
        return (p[0]/self.scansize[0]*self.pixelsize[0], p[1]/self.scansize[1]*self.pixelsize[1])
            
    def pixels_to_nm(self, p):
        """converts x,y coordinates from pixel to nm coordinates"""
        return (p[0]*self.scansize[0]/self.pixelsize[0], p[1]*self.scansize[1]/self.pixelsize[1])

    
def get_distance(p1,p2):
    """returns the Euclidean distance between two points"""
    return np.sqrt(np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))

    
def drifted_point(p, drift, elapsed_time):
    """returns the corrected coordinates of point p (any unit) taking into account drift (unit per time) and the elapsed time (time-units)"""
    return (p[0] + drift[0]*elapsed_time, p[1] + drift[1]*elapsed_time)

    
def string_simplify(str):
    """simplifies a string (i.e. removes replaces space for "_", and makes it lowercase"""
    return str.replace(' ','_').lower()




if __name__=="__main__":
    pass
