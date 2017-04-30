import matplotlib.colors

cdict = {'red':   ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),(1.0, 1.0, 1.0),)}
greys_linear = matplotlib.colors.LinearSegmentedColormap('greys_linear', cdict)  # always have trouble with the brightness values


# Alex's custom colormaps
cdict_BlueA = {'red':   ((0.0,  0.0, 0.0),(0.25,  0.094, 0.094),(0.67,  0.353, 0.353),(1.0,  1.0, 1.0)),
               'green': ((0.0,  0.0, 0.0),(0.25, 0.137, 0.137),(0.67, 0.537, 0.537),(1.0,  1.0, 1.0)),
               'blue':  ((0.0,  0.0, 0.0),(0.25, 0.2, 0.2),(0.67, 0.749, 0.749),(1.0,  1.0, 1.0))}
         
# little brighter version of BlueA
cdict_BlueAb = {'red':   ((0.0,  0.0, 0.0),(0.22,  0.094, 0.094),(0.60,  0.353, 0.353),(1.0,  1.0, 1.0)),
                'green': ((0.0,  0.0, 0.0),(0.22, 0.137, 0.137),(0.60, 0.537, 0.537),(1.0,  1.0, 1.0)),
                'blue':  ((0.0,  0.0, 0.0),(0.22, 0.2, 0.2),(0.60, 0.749, 0.749),(1.0,  1.0, 1.0))}
cdict_BlueA2 = {'red':   ((0.0,  0.0, 0.0), (0.25,  0.055, 0.055),(0.67,  0.212, 0.212),(1.0,  1.0, 1.0)),
                'green': ((0.0,  0.0, 0.0),(0.25, 0.106, 0.106),(0.67, 0.455, 0.455),(1.0,  1.0, 1.0)),
                'blue':  ((0.0,  0.0, 0.0),(0.25, 0.231, 0.231),(0.67, 0.749, 0.749),(1.0,  1.0, 1.0))}

BlueA = matplotlib.colors.LinearSegmentedColormap('BlueA', cdict_BlueA)
BlueAb = matplotlib.colors.LinearSegmentedColormap('BlueAb', cdict_BlueAb)
BlueA2 = matplotlib.colors.LinearSegmentedColormap('BlueA2', cdict_BlueA2)