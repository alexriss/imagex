"""
imagex

SPM image import and analysis

Class to read Scanning Probe Microscopy data (Nanonis sxm files, as well as some forms of Createc dat files) and do some analysis.


2017, Alex Riss, GPL

Parts of the load-file function are based on code from the Nanonis manual, written by Felix Wählisch (Beer-ware license).

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import imagex.colormap as cm
import ipywidgets
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import re
import scipy.ndimage.interpolation
import skimage.measure
import struct
import sys
# import zlib  # will be imported in the function when it is needed, used only to read compressed CreaTec files

import seaborn as sns
sns.set_style("dark")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.5})
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

PYTHON_VERSION = sys.version_info.major



class ImageData(object):
    """Class to read grid a Nanonis sxm image and analyze it. """

    def __init__(self):
        self.filename = ""         # filename
        self.header = {}           # dictionary containing all the header data
        self.channels = []         # list holding a dictionary with data_header (names, units and the swept channels) and data (np.array of the actual data)
        self.channel_names = []    # list holding the channel names

        self.scansize = [0,0]      # scan size
        self.scansize_unit = ""    # unit for scansize (e.g. nm)
        self.pixelsize = [0,0]     # scan size in pixels
        self.scan_direction = ""   # scan direction ("up" or "down")

        self.start_time = datetime.date.today() # start time of scan
        self.acquisition_time = 0   # aquisition time in seconds


    def load_image(self,fname,output_info=1):
        """Loads SPM data from a Nanonis sxm or Createc dat file.
        Args:
            fname (str): Filename of the data file to read.
            output_info (int): Specifies the amount of output info to print to stdout when reading the files. 0 for no output, 1 for limited output, 2 for detailed output.

        Raises:
            NotImplementedError: When the file extension is not known.
        """
        ext = fname.rsplit(".",1)[-1]
        if  ext == "sxm":
            self.__init__()
            self._load_image_header_nanonis(fname,output_info)
            self._load_image_body_nanonis(fname,output_info)
        elif ext == 'dat':
            self.__init__()
            self._load_image_header_createc(fname,output_info)
            self._load_image_body_createc(fname,output_info)
        else:
            raise NotImplementedError('Unknown file type \"%s\".' % ext)
        self.filename = fname


    def get_channel(self, channel_name):
        """Gets the channel dictionary for channel_name
        Args:
            channel_name (str): string specifying the channel name to return

        Returns:
            dict: dictionary containing the information and date corresponding to channel_name.

        Raises:
            ValueError: When channel_name is not found.
        """
        if channel_name not in self.channel_names:
            if channel_name[-4:] not in ("_fwd", "_bwd"):
                channel_name += '_fwd'
                print('Using channel %s' % channel_name)
            if channel_name not in self.channel_names:
                print('Error: Channel %s does not exist in %s.' % (channel_name, self.filename))
                print('    Available channels are: %s.' % ", ".join(self.channel_names))
                raise ValueError('Channel %s not found.' % channel_name)

        i_channel = self.channel_names.index(channel_name)
        return self.channels[i_channel]


    def get_data(self, channel_name):
        """Gets the channel data for channel_name
        Args:
            channel_name (str): string specifying the channel name to return

        Returns:
            data (numpy array): The image data in a 2D (row/column = y,x) format.
        """
        
        return self.get_channel(channel_name)['data']
    

    def plot_image(self, channel_name, cmap='', interpolation="bicubic", background="none", no_labels=False, pixel_units=False, output_filename="", output_dpi=100, alpha=1, axes=None, extra_output=True, **kwargs):
        """Plots channel data.

        Args:
            channel_name (str): name of the channel to plot.
            cmap: Matplotlib colormap to use (see the cmap-parameter in matplotlib.axes.Axes.imshow for more information).
            interpolation (str): Interpolation type (see the interpolation-parameter in matplotlib.axes.Axes.imshow for more information).
            background (str): If set to 'offset', an offset will be calculated, such that the lowest value in the 2D dataset is zero.
            no_labels (bool): Specifies whether axes labels will be used. The labels will be the channel name and the channel unit.
            pixel_units: If True, then the x and y axis will be plotted in pixel_units (instead of nm).
            output_filename (str): If output_filename is given, then the plot is saved under this filename with resolution output_dpi.
            output_dpi (float): Resolution for saving images.
            alpha: Opacity value to pass to the matplotlib.axes.Axes.imshow function.
            extra_output: Specifies whether detailed output should be printed to stdout.
            axes: Matplotlib axes object to use for plot output. If no axes is given, a new matplotlib Figure object will be created for the plot.
            **kwargs: Any additional keyword arguments to be passed to matplotlib.axes.Axes.imshow.

        Returns:
            matplotlib figure: If no axes is given,  a new matplotlib Figure object will be created for the plot and returned.
        """
        channel = self.get_channel(channel_name)
        if channel is False: return False

        # z = np.fliplr(np.rot90(self.channels[i_channel]['data'], k=1))
        z = channel['data']
        unit = channel['data_header']['unit']
        name = channel['data_header']['name']
        return self.plot_data(z, name=name, unit=unit, cmap=cmap, interpolation=interpolation, background=background, no_labels=no_labels, pixel_units=pixel_units, output_filename=output_filename, output_dpi=output_dpi, alpha=alpha, axes=axes, extra_output=extra_output, **kwargs)


    def plot_data(self, data, name="", unit="", cmap='', interpolation="bicubic", background="none", no_labels=False, pixel_units=False, output_filename="", output_dpi=100, alpha=1, axes=None, extra_output=True, **kwargs):
        """Plots 2D data.

        Args:
            data (numpy array): 2D data in row/column=y,x format.
            name (str): Specifies the axes title.
            unit (str): Units to be appended to the axes title.
            cmap: Matplotlib colormap to use (see the cmap-parameter in matplotlib.axes.Axes.imshow for more information).
            interpolation (str): Interpolation type (see the interpolation-parameter in matplotlib.axes.Axes.imshow for more information).
            background (str): If set to 'offset', an offset will be calculated, such that the lowest value in the 2D dataset is zero.
            no_labels (bool): Specifies whether axes labels will be used.
            pixel_units: If True, then the x and y axis will be plotted in pixel_units (instead of nm).
            output_filename (str): If output_filename is given, then the plot is saved under this filename with resolution output_dpi.
            output_dpi (float): Resolution for saving images.
            alpha: Opacity value to pass to the matplotlib.axes.Axes.imshow function.
            extra_output: Specifies whether detailed output should be printed to stdout.
            axes: Matplotlib axes object to use for plot output. If no axes is given, a new matplotlib Figure object will be created for the plot.
            **kwargs: Any additional keyword arguments to be passed to matplotlib.axes.Axes.imshow.

        Returns:
            matplotlib figure: If no axes is given,  a new matplotlib Figure object will be created for the plot and returned.
        """

        if not pixel_units: x_len, y_len = self.scansize[0], self.scansize[1]
        z = data
        # if unit == 'm':
            # z = z*1.0e9
            # unit = 'n'+unit
        if background == "offset":
            z_orig = z
            z = z - np.amin(z)

        if cmap=='': cmap=cm.greys_linear

        if axes == None:
            fig = plt.figure()
            if no_labels:
                ax = fig.add_axes([0,0,1,1])
            else:
                ax = plt.subplot(111)
        else:
            ax = axes
            for im in ax.images:
                if im.colorbar: im.colorbar.remove()  # remove any present colorbars
            plt.sca(ax)
        img = ax.imshow(z, cmap=cmap, aspect='equal', interpolation=interpolation, origin='lower', picker=True, alpha=alpha, **kwargs)
        if not pixel_units: plt.setp(img, extent=(0,x_len,0,y_len))

        ax.grid(False)
        if no_labels:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            if pixel_units:
                ax.set_xlabel("X [px]")
                ax.set_ylabel("Y [px]")
            else:
                ax.set_xlabel("X [nm]")
                ax.set_ylabel("Y [nm]")

            fig_title = name
            if unit: fig_title += ' ['+unit+']'
            if background != "none":
                fig_title += "; background correction: %s" % background
            ax.set_title(fig_title)

            #divider = mpl_toolkits.axes_grid1.make_axes_locatable(plt.gca())
            #cax = divider.append_axes("right", "5%", pad="3%")
            #cbar = plt.colorbar(img, cax=cax)
            #cbar.ax.tick_params(labelsize=8)

            cbar = plt.colorbar(img, fraction=0.046, pad=0.04) # fraction=0.046, pad=0.04, shrink=0.8
            #cbar.set_label(name+' ['+unit+']')

        if extra_output: print("%s: min=%s%s, max=%s%s" % (name, np.amin(z), unit, np.amax(z), unit))
        if background != "none":
            if extra_output: print("%s (without background correction): min=%s%s, max=%s%s" % (name, np.amin(z_orig), unit, np.amax(z_orig), unit))
        if output_filename != "":
            plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=output_dpi)
            if extra_output: print("Saved: %s" % output_filename)

        if axes == None: return fig


    def line_profile(self, data, points=[], linewidth=1, outside_range='constant', **kwargs):
        """Calculates a line profile along the points array (x,y coordinates in nm units) of certain width.

        Args:
            data (numpy array): The image data in a 2D (row/column = y,x) format.
            points: List of points in nm coordinates (x,y), fow which the line profile should be calculated.
            linewidth (float): Linewidth in pixel values.
            outside_range: Specifies how to treat potential values outside of the input data array. See the mode-parameter in skimage.measure.profile_line.
            **kwargs: additional keyword arguments to be passed to skimage.measure.profile_line.

        Returns:
            lines: np array of positions and corresponding image-values. If more than 2 points are given,m a list of lines is returned corresponding to the line-data for the sections between the points.

        """
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
            y = skimage.measure.profile_line(data, p1, p2, linewidth=linewidth, mode=outside_range, **kwargs)
            line_length = get_distance(points[i], points[i+1])
            x = np.linspace(x_pos, line_length, len(y))
            x_pos += line_length
            lines.append(np.array([x,y]))

        if len(lines) == 1:
            return lines[0]
        else:
            return lines


    def subtract_plane(self, data, points_plane, interpolation_order=1):
        """Performs plane subtraction from the data

        Args:
            data: numpy array containing the original 2d data
            points_plane: x,y coordinates spanning the plane (col first, then row)
            interpolation_order: spline interpolation order for picking values from original array

        Returns:
            numpy array: plane-subtracted data
        """

        points_plane_px = np.array([self.nm_to_pixels(p) for p in points_plane])
        col_px = points_plane_px[:,0]
        row_px = points_plane_px[:,1]
        z = scipy.ndimage.interpolation.map_coordinates(data, [row_px,col_px], order=interpolation_order, mode='nearest')

        # regression
        A = np.c_[row_px, col_px, np.ones(row_px.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, z)

        # evaluate it on grid
        data_indices = np.indices(data.shape)

        zz = C[0]*data_indices[0] + C[1]*data_indices[1] + C[2]
        return data-zz


    def super_lattice(self, data, lattice_vectors, origin, output_size, interpolation_order=1):
        """Creates super lattice by repeating image according to given lattice vectors

        Args:
            data: numpy array containing the original 2d data
            lattice_vectors: lattice vectros in 2d that specify the translational symmetry
            origin: origin of
            output_size: output_size of generated image in nm
            interpolation: spline interpolation order for picking values from original array

            The parameters lattice_vectors, origin, output_size are specified in x,y format, i.e. col first, then row.

        Returns:
            numpy array: new stitched image
        """

        super_data_size = np.rint(np.array(self.nm_to_pixels(output_size[::-1]))).astype(np.int)  # the [::-1] transforms from x,y into row, col format
        super_data_interpolation_indices = np.indices(super_data_size)

        lattice_vectors_px = np.array([self.nm_to_pixels(lattice_vectors[0][::-1]), self.nm_to_pixels(lattice_vectors[1][::-1])])  # the [::-1] transforms from x,y into row, col format
        origin_px = np.array(self.nm_to_pixels(origin[::-1]))  # the [::-1] transforms from x,y into row, col format

        for j in range(0, super_data_size[0]):
            for i in range(0, super_data_size[1]):
                lattice_coord = np.matmul(np.linalg.inv(lattice_vectors_px).T, np.array([j,i]).T).T % 1    # the %1 ensures we are always between 0 and 1
                px_coord = np.matmul(lattice_vectors_px.T, lattice_coord)
                super_data_interpolation_indices[0,j,i] = px_coord[0] + origin_px[0]
                super_data_interpolation_indices[1,j,i] = px_coord[1] + origin_px[1]

        return scipy.ndimage.interpolation.map_coordinates(data, super_data_interpolation_indices, order=interpolation_order, mode='nearest')


    def _load_image_header_nanonis(self,fname,output_info):
        """Loads header data from a .sxm file

        Args:
            fname (str): Filename of the data file to read.
            output_info (int): Specifies the amount of output info to print to stdout when reading the files. 0 for no output, 1 for limited output, 2 for detailed output.
        """

        if output_info>0: print('Reading header of %s' % fname)
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
        self.scansize = [float(x_len)*1e9, float(y_len)*1e9]  # convert to nm
        self.scansize_unit = 'nm'
        xPixels, yPixels = self.header['scan_pixels'].split()
        self.pixelsize = [int(xPixels), int(yPixels)]
        self.scan_direction = self.header['scan_dir']

        self.start_time = datetime.datetime.strptime(self.header['rec_date'] + " " + self.header['rec_time'], '%d.%m.%Y %H:%M:%S')
        self.acquisition_time = float(self.header['acq_time'])


    def _load_image_body_nanonis(self,fname,output_info):
        """Loads body data from a .sxm file.

        Args:
            fname (str): Filename of the data file to read.
            output_info (int): Specifies the amount of output info to print to stdout when reading the files. 0 for no output, 1 for limited output, 2 for detailed output.
        """
        # extract channels to be read in
        if output_info>0: print('Reading body of %s' % fname)
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
            if output_info>1:
                print("  read: %s in %s, shape: %s" % channel['data_header'].name, channel['data_header'].unit, channel['data'].shape)
            self.channels.append(channel)
            self.channel_names.append(channel['data_header']['name'])
        f.close()


    def _load_image_header_createc(self,fname,output_info):
        """Loads header data from a createc .dat file

        Args:
            fname (str): Filename of the data file to read.
            output_info (int): Specifies the amount of output info to print to stdout when reading the files. 0 for no output, 1 for limited output, 2 for detailed output.

        Raises:
            NotImplementedError: When the file extension is not known.  // [todo]
        """
        if output_info>0: print('Reading header of %s' % fname)
        if PYTHON_VERSION>=3:
            f = open(fname, encoding='utf-8', errors='ignore')
        else:
            f = open(fname)
        header_ended = False
        caption = re.compile(':*:')
        key = ''
        contents = ''
        firstline = f.readline().strip()
        self.fileformat = firstline
        while not header_ended:
            line = f.readline()
            if not line: break
            if line[0:11] == "PSTMAFM.EXE":  # end of header
                header_ended = True
                self.header[key] = contents
            else:
                parts = line.split('=')
                if len(parts)!=2: continue
                key, contents = parts
                line = line.strip()
                key = string_simplify(key)  # set new name
                self.header[key] = contents.strip()
                # [todo: add some parsing here]
        f.close()

        x_len, y_len = self.header['length_x[a]'], self.header['length_y[a]']
        self.scansize = [float(x_len)/10, float(y_len)/10]  # convert to nm
        self.scansize_unit = 'nm'
        xPixels, yPixels = self.header['num.x_/_num.x'], self.header['num.y_/_num.y']
        self.pixelsize = [int(xPixels), int(yPixels)]
        self.scan_direction = 'down'

        # todo: get these values if possible
        #self.start_time = datetime.datetime.strptime(self.header['rec_date'] + " " + self.header['rec_time'], '%d.%m.%Y %H:%M:%S')
        #self.acquisition_time = float(self.header['acq_time'])


    def _load_image_body_createc(self,fname,output_info):
        """Loads body data from a createc .dat file

        Args:
            fname (str): Filename of the data file to read.
            output_info (int): Specifies the amount of output info to print to stdout when reading the files. 0 for no output, 1 for limited output, 2 for detailed output.

        Raises:
            NotImplementedError: When the file extension is not known.  // [todo]
        """
        if not self.fileformat in ["[Paramco32]", "[Paramet32]"]: # 32 bit compressed, or uncompressed
            raise NotImplementedError('Reading of data in files saved in %s- format is not implemented yet.' % self.fileformat)

        # extract channels to be read in
        if output_info>0: print('Reading body of %s' % fname)
        xPixels, yPixels = self.pixelsize

        z_conv = float(self.header['dacto[a]z'])  # conversion for z scale
        z_gain = float(self.header['gainz_/_gainz']) # gain

        preamp_fac = 10**float(self.header['gainpreamp_/_gainpre_10^'])
        z_piezo_const = float(self.header['zpiezoconst'])

        num_chan = int(self.header['channels_/_channels']) # number of channels
        config_chan = int(self.header['chan(1,2,4)_/_chan(1,2,4)']) # configuration channels 1: topo, 2: topo+current, 4: topo+current+adc1+adc2

        names_temp = ['z_fwd','current_fwd','adc1_fwd','adc2_fwd']
        units_temp = ['nm','nA.','mV','mV']  # I am not completely sure about the current conversion, also I am not sure about the mV of adc1 and adc2.
        names = []
        units = []
        for j in range(config_chan):
            names.append(names_temp[j])
            units.append(units_temp[j])
        if num_chan == 2*config_chan:  # there are backwards scans
            names_back = [n.replace('_fwd', '_bwd') for n in names]
            names = names + names_back
            units = units + units
        f = open(fname, 'rb') #read binary
        read_all = f.read()
        offset = read_all.find(bytearray('DATA', encoding='ascii'))
        f.seek(offset+4)  # data start 6 bytes afterwards
        read_all = f.read()

        if self.fileformat == "[Paramco32]":
            import zlib
            data_array = zlib.decompress(read_all)
        elif self.fileformat == "[Paramet32]":
            data_array = read_all
        fmt = '<f' # float
        ItemSize = struct.calcsize(fmt)
        extra_offset = 4

        for i in range(len(names)):
            data = np.zeros(xPixels*yPixels)
            for j in range(xPixels*yPixels):
                data[j] = struct.unpack(fmt,
                    data_array[extra_offset + i*ItemSize*xPixels*yPixels + j*ItemSize:
                    extra_offset + i*ItemSize*xPixels*yPixels + j*ItemSize+ItemSize])[0]
            data = data.reshape(yPixels, xPixels)

            # lets see if we need it
            if self.scan_direction == "down":
                data = np.flipud(data)
            if names[i][0] == 'z':
                data = data * z_conv * z_gain / 10  # convert to nm
            elif names[i][0] == 'c':
                data = data / preamp_fac / z_piezo_const * z_conv
            channel = {'data_header': {'name': names[i], 'unit': units[i]}, 'data': data}

            if output_info>1:
                print("  read: %s in %s, shape: %s" % channel['data_header'].name, channel['data_header'].unit, channel['data'].shape)
            self.channels.append(channel)
            self.channel_names.append(channel['data_header']['name'])
        f.close()


    def nm_to_pixels(self, p):
        """Converts from nm to pixel coordinates. Input is either a single number or x,y coordinates.

        Args:
            p: Either a single number (int, float) or xy coordinates (list, tuple, numpy array)

        Returns:
            float or list: The converted number or xy coordinates.

        Raises:
            ValueError: If the input coordinates do not correspond to the format specified above.
        """
        if isinstance(p, (int, float)):
            return p/self.scansize[0]*self.pixelsize[0]
        elif isinstance(p, (list, tuple, np.ndarray)) and len(p)==2:
            return [p[0]/self.scansize[0]*self.pixelsize[0], p[1]/self.scansize[1]*self.pixelsize[1]]
        else:
            raise ValueError('The input to nm_to_pixels should either be a single number or x,y coordinates.')


    def pixels_to_nm(self, p):
        """Converts from pixel to nm coordinates. Input is either a single number or x,y coordinates.

        Args:
            p: Either a single number (int, float) or xy coordinates (list, tuple, numpy array)

        Returns:
            float or list: The converted number or xy coordinates.

        Raises:
            ValueError: If the input coordinates do not correspond to the format specified above.
        """
        if isinstance(p, (int, float)):
            return p*self.scansize[0]/self.pixelsize[0]
        elif isinstance(p, (list, tuple, np.ndarray)) and len(p)==2:
            return [p[0]*self.scansize[0]/self.pixelsize[0], p[1]*self.scansize[1]/self.pixelsize[1]]
        else:
            raise ValueError('The input to pixels_to_nm should either be a single number or x,y coordinates.')


def save_image(filename, data, cmap=cm.greys_linear, **kwargs):
    """Saves 2D image data as an image to a file.

    Args:
        filename: A string containing a path to a filename, or a Python file-like object.
        data (numpy array): The image data in a 2D (row/column = y,x) format.
        **kwargs: Additional kwargs to be passed to matplotlib.image.imsave.
    """
    matplotlib.image.imsave(filename, data, cmap=cmap,  **kwargs)


def save_axes(axes, filename, dpi=100, pad_inches=0, transparent=True, **kwargs):
    """Saves matplotlib axes object to a graphics file.

    Args:
        axex: Matplotlib axes object.
        filename: A string containing a path to a filename, or a Python file-like object.
        dpi: Resolution in dots per inch for the output file.
        pad_inches: see matplotlib.pyplot.savefig.
        bbox_inches: see matplotlib.pyplot.savefig.
        transparent: see matplotlib.pyplot.savefig.
        figsize: tuple specifying the width and height in inches.
        **kwargs: Additional kwargs to be passed to matplotlib.pyplot.savefig.
    """
    fig = axes.get_figure()
    extent = axes.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=dpi, bbox_inches=extent, facecolor=fig.get_facecolor(), transparent=transparent, **kwargs)    

    
def save_figure(fig, filename, dpi=100, pad_inches=0, bbox_inches="tight", transparent=True, figsize=(), **kwargs):
    """Saves matplotlib figure object to a graphics file.

    Args:
        fig: Matplotlib figure object.
        filename: A string containing a path to a filename, or a Python file-like object.
        dpi: Resolution in dots per inch for the output file.
        pad_inches: see matplotlib.pyplot.savefig.
        bbox_inches: see matplotlib.pyplot.savefig.
        transparent: see matplotlib.pyplot.savefig.
        figsize: tuple specifying the width and height in inches.
        **kwargs: Additional kwargs to be passed to matplotlib.pyplot.savefig.
    """
    if len(figsize)==2:
        figsize_orig = fig.get_size_inches()
        fig.set_size_inches(figsize)
    fig.savefig(filename, dpi=dpi, pad_inches=pad_inches, bbox_inches=bbox_inches, transparent=transparent, **kwargs)
    if len(figsize)==2:
        fig.set_size_inches(figsize_orig)


def images_colorscale(axs_list, min, max):
    """Sets the color scale for the images in the axes.
    Args:
        axs_list: List of matplotlib axes objects containing images.
        min (float): Minimum value of color scale.
        max (float): Maximum value of color scale.
    """
    for ax in axs_list:
        for im in ax.images:
            im.set_clim(vmin=min, vmax=max)
            im.cmap.set_under('#0000ff')
            im.cmap.set_over('#ff0000')


def images_colorscale_sliders(axs_list, scale_step_size=0, display=True):
    """Creates ipywidget sliders to scale the 0th image for each axes in axs_list.

    Args:
        axs_list: List of matplotlib axes objects containing images.
        scale_step_size: The step size for the sliders to use. If set to 0 (default), then an automatic step_size will be picked.
        display: Specifies whether the ipywsliders will be displayed or returned.

    Returns:
        ipywidgets object: If display is False. If display is True, the widgets will be directly diplayed.
    """
    import IPython.display
    
    ias = []
    for ax in axs_list:
        scale_min, scale_max = ax.images[0].get_clim()
        if scale_step_size == 0:
            scale_step_size_curr = np.abs(scale_min-scale_max)/100
            scale_step_size_curr = 10**(np.floor(np.log10(scale_step_size_curr)))
        ia = ipywidgets.interactive(images_colorscale,
                    min=ipywidgets.widgets.FloatSlider(min=scale_min,max=scale_max+scale_step_size_curr,step=scale_step_size_curr,value=scale_min),
                    max=ipywidgets.widgets.FloatSlider(min=scale_min,max=scale_max+scale_step_size_curr,step=scale_step_size_curr,value=scale_max),
                    axs_list=ipywidgets.widgets.fixed([ax]))
        ias.append(ia)
    if display:
        IPython.display.display(ipywidgets.HBox(ias))
    else:
        return ias


def get_distance(p1,p2):
    """Calculates the Euclidean distance between two points

    Args:
        p1, p2: Iterables of length n.

    Returns:
        int: Calculated distance between p1 and p2 in n-dimensional space.
    """
    return np.sqrt(np.sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))


def drifted_point(p, drift, elapsed_time):
    """Calculates the corrected coordinates of point p (any unit) taking into account drift (unit per time) and the elapsed time (time-units)

    Args:
        p: Iterable of length 2 specifying the point coordinates in the xy plane.
        drift (float): Value of sptial drift in length units per time unit.
        elapsed_time (float): Elapsed time for which the drift correction should be calculated.

    Returns:
        tuple: Drift-corrected coordinates of point p.
    """
    return (p[0] + drift[0]*elapsed_time, p[1] + drift[1]*elapsed_time)


def string_simplify(str):
    """Simplifies a string (i.e. removes replaces space for "_", and makes it lowercase

    Args:
        str (str): Input string.

    Returns:
        str: Simplified output string.
    """
    return str.replace(' ','_').lower()




if __name__=="__main__":
    pass
