# Python translation of surface analysis code
# Original Code - Jan 2026 - Arash Nikniazi
# Translation - Feb 2026 - Daniel Howells

import numpy as np
from scipy.interpolate import griddata
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def azar_surface_analysis(sample_data, resolution, pixel_width, rot90_value=0, figure_on=False):
    """
    Calculate HD, AH and SS of Z map
    
    Parameters:
    -----------
    sample_data : ndarray
        Raw data from LDD700 Output with columns [X, Y, Z, I]
    resolution : int
        Resolution of the height map
    pixel_width : float
        Physical width of each pixel
    rot90_value : int
        Rotates Height Map counterclockwise by rot90_value*90 degrees
    figure_on : bool
        If True, plots will be generated
        
    Returns:
    --------
    HD : float
        Height Difference - mean height offset between contour-dominated 
        region and interior hatch region
    AH : float
        Average Height of FFT region
    SS : float
        Surface Smoothness - area under power spectrum density curve

    """
    
    # Parameters
    HD_INNER_SIZE = 800
    HD_EDGE_INDEX = 1700
    HD_EDGE_SIZE = 200
    INTER_RANGE = 1500
    
    ## Formatting the Height Map
    
    X = sample_data[:, 0].reshape(-1, resolution)
    X = X[:, 0]
    Z = sample_data[:, 2].reshape(-1, resolution)
    I = sample_data[:, 3].reshape(-1, resolution)
    
    # Flip every other column
    for i in range(resolution):
        if (i + 1) % 2 == 0:  # Adjust for MATLAB's 1-based indexing
            Z[:, i] = np.flip(Z[:, i])
            I[:, i] = np.flip(I[:, i])
    
    Z_mean = np.nanmean(Z)
    
    lowbound = Z_mean - INTER_RANGE + 500
    upbound = Z_mean + INTER_RANGE
    
    Z[Z < lowbound] = np.nan
    Z[Z > upbound] = np.nan
    
    # Fill missing values using moving median
    Z_inter = fill_missing_movmedian(Z, window=50)
    
    if rot90_value != 0:
        Z_inter = np.rot90(Z_inter, rot90_value)
    
    ## Fitting - subtract 2D surface to remove large-scale tilt/curvature
    
    N_y, N_x = Z_inter.shape
    xGrid, yGrid = np.meshgrid(np.arange(1, N_x + 1), np.arange(1, N_y + 1))
    
    # Fit 2nd order polynomial surface
    Z_fit = fit_poly_surface_2d(xGrid, yGrid, Z_inter, order=2)
    Z_inter_detrend = Z_inter - Z_fit
    
    # Replace NaN with minimum value
    Z_inter_detrend[np.isnan(Z_inter_detrend)] = np.nanmin(Z_inter_detrend)
    
    ## Height Difference (HD) Calculation
    
    HD_Z_area = np.zeros((len(Z), len(Z)))
    
    Z_cross_section = np.nanmedian(Z_inter_detrend, axis=0)
    
    X_edge = X[HD_EDGE_INDEX:HD_EDGE_INDEX + HD_EDGE_SIZE]
    Z_edge = Z_cross_section[HD_EDGE_INDEX:HD_EDGE_INDEX + HD_EDGE_SIZE]
    HD_Z_area[:, HD_EDGE_INDEX:HD_EDGE_INDEX + HD_EDGE_SIZE] = 2
    Z_edge_mean = np.max(Z_edge)
    
    X_inside = X[0:HD_INNER_SIZE]
    Z_inside = Z_cross_section[0:HD_INNER_SIZE]
    HD_Z_area[resolution - HD_INNER_SIZE:, 0:HD_INNER_SIZE] = 1
    Z_inside_mean = np.median(Z_inside)
    
    height_diff = Z_edge_mean - Z_inside_mean
    HD = height_diff
    
    ## Average Height (AH) Calculation
    
    SS_region = round(0.8 * resolution)
    
    X_inside_all = X[0:SS_region]
    Z_inside_all = Z_cross_section[0:SS_region]
    Z_inside_all_mean = np.median(Z_inside_all)
    
    AH = Z_inside_all_mean
    
    ## Surface Smoothness (SS) Calculation
    
    SS_Z_area = np.zeros((len(Z), len(Z)))
    
    Z_SS = Z_inter_detrend[resolution - SS_region:, 0:SS_region]
    SS_Z_area[resolution - SS_region:, 0:SS_region] = 1
    
    # FFT with zero-padding to next power of 2
    n_fft_y = 2 ** int(np.ceil(np.log2(Z_SS.shape[0])))
    n_fft_x = 2 ** int(np.ceil(np.log2(Z_SS.shape[1])))
    
    Zf = fft2(Z_SS, s=(n_fft_y, n_fft_x))
    PSD_NA1 = np.abs(Zf) ** 2
    PSD_NA1 = fftshift(PSD_NA1)  # Center low frequencies
    
    dx = pixel_width
    fx = np.fft.fftshift(np.fft.fftfreq(PSD_NA1.shape[1], dx))
    fy = np.fft.fftshift(np.fft.fftfreq(PSD_NA1.shape[0], dx))
    fx, fy = np.meshgrid(fx, fy)
    
    fr = np.sqrt(fx**2 + fy**2)  # radial frequency
    lambda_val = 1.0 / (fr + np.finfo(float).eps)  # spatial wavelength
    
    lambda_ranges = [20e-6, 200e-6]  # [λ_min λ_max] per band
    
    mask = (lambda_val >= lambda_ranges[0]) & (lambda_val <= lambda_ranges[1])
    
    SS_NA1 = np.sum(PSD_NA1[mask])
    SS = SS_NA1
    
    ## Plots
    
    if figure_on:
        fig = plt.figure(figsize=(12, 10))
        
        # Subplot 1: Z Inter Detrend
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(Z_inter_detrend, extent=[X[0], X[-1], X[-1], X[0]], 
                         aspect='equal', cmap='jet')
        ax1.set_title('Z Inter Detrend')
        ax1.set_xlabel('X Coordinate (μm)')
        ax1.set_ylabel('Y Coordinate (μm)')
        plt.colorbar(im1, ax=ax1)
        
        # Subplot 2: Height Difference Area
        ax2 = plt.subplot(2, 2, 2)
        overlay_image(ax2, Z_inter_detrend, HD_Z_area)
        ax2.set_title('Height Difference Area')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_aspect('equal')
        
        # Subplot 3: Surface Smoothness Area
        ax3 = plt.subplot(2, 2, 3)
        overlay_image(ax3, Z_inter_detrend, SS_Z_area)
        ax3.set_title('Surface Smoothness Area')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_aspect('equal')
        
        # Subplot 4: Cross Section
        ax4 = plt.subplot(2, 2, 4)
        ax4_right = ax4.twinx()
        
        ax4.plot(X, np.nanmedian(Z, axis=1), label='Raw')
        ax4.set_ylabel('Raw Height Map (μm)', color='C0')
        ax4.tick_params(axis='y', labelcolor='C0')
        
        ax4_right.plot(X, Z_cross_section, 'C1', label='INP & DET')
        ax4_right.plot(X_edge, Z_edge, 'm-', linewidth=2)
        ax4_right.plot(X_inside_all, Z_inside_all, 'g-', linewidth=2)
        ax4_right.plot(X_inside, Z_inside, 'b-', linewidth=2)
        ax4_right.set_ylabel('Interpolate Height Map (μm)', color='C1')
        ax4_right.tick_params(axis='y', labelcolor='C1')
        
        ax4_right.legend(['Raw', 'INP & DET'], loc='upper left')
        ax4.set_title('Cross Section')
        ax4.set_xlabel('X Coordinate (μm)')
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    return HD, AH, SS


def fill_missing_movmedian(Z, window=50):
    """Fill missing (NaN) values using moving median"""
    Z_filled = Z.copy()
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isnan(Z[i, j]):
                # Define window bounds
                i_min = max(0, i - window // 2)
                i_max = min(Z.shape[0], i + window // 2 + 1)
                j_min = max(0, j - window // 2)
                j_max = min(Z.shape[1], j + window // 2 + 1)
                
                # Get window values
                window_vals = Z[i_min:i_max, j_min:j_max]
                
                # Fill with median of non-NaN values
                if not np.all(np.isnan(window_vals)):
                    Z_filled[i, j] = np.nanmedian(window_vals)
    
    return Z_filled


def fit_poly_surface_2d(xGrid, yGrid, Z, order=2):
    """Fit a 2D polynomial surface to data"""
    # Flatten the grids and Z
    x_flat = xGrid.flatten()
    y_flat = yGrid.flatten()
    z_flat = Z.flatten()
    
    # Remove NaN values
    valid = ~np.isnan(z_flat)
    x_valid = x_flat[valid]
    y_valid = y_flat[valid]
    z_valid = z_flat[valid]
    
    # Build design matrix for 2nd order polynomial
    # z = a0 + a1*x + a2*y + a3*x^2 + a4*xy + a5*y^2
    A = np.column_stack([
        np.ones_like(x_valid),
        x_valid, y_valid,
        x_valid**2, x_valid * y_valid, y_valid**2
    ])
    
    # Solve least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
    
    # Evaluate polynomial on full grid
    A_full = np.column_stack([
        np.ones_like(x_flat),
        x_flat, y_flat,
        x_flat**2, x_flat * y_flat, y_flat**2
    ])
    
    z_fit = A_full @ coeffs
    
    return z_fit.reshape(Z.shape)


def overlay_image(ax, base_image, mask):
    """Create overlay visualization similar to MATLAB's imshowpair"""
    # Normalize base image
    base_norm = (base_image - np.nanmin(base_image)) / (np.nanmax(base_image) - np.nanmin(base_image))
    
    # Create RGB image
    rgb = np.zeros((*base_image.shape, 3))
    rgb[:, :, 0] = base_norm  # Red channel
    rgb[:, :, 1] = base_norm  # Green channel
    rgb[:, :, 2] = base_norm  # Blue channel
    
    # Overlay mask in magenta/cyan
    rgb[:, :, 0] = np.where(mask == 1, 1, rgb[:, :, 0])  # Region 1 in red
    rgb[:, :, 1] = np.where(mask == 2, 1, rgb[:, :, 1])  # Region 2 in green
    rgb[:, :, 2] = np.where(mask == 2, 1, rgb[:, :, 2])  # Region 2 in cyan-ish
    
    ax.imshow(rgb)