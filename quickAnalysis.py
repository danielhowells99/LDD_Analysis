from analysis import azar_surface_analysis
import os
import numpy as np

RESOLUTION = 2000

def analyze_file(filepath):
    with open(filepath, 'r') as f:
        data = np.loadtxt(f,delimiter = ',')
        HD, AH, SS = azar_surface_analysis(data, resolution=RESOLUTION, pixel_width=1e-6, rot90_value=0, figure_on=False)
        return HD, AH, SS
        
analyze_file('TestData20002000.txt')