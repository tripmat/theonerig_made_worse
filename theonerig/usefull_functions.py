__all__ = ['get_overlap', "swarmplot_x",'sum_of_gaussians_ChatGPT','fit_sum_of_gaussians', 
           'generate_ticks', "find_spatial_STA_center", "exponentiated_sine_wave_model",
           "evaluate_fit_of_exponentiated_sine_wave_model", "doeModelInt", "getTFFitLsq"]

import h5py
import json, re
import numpy as np
from collections import namedtuple
from typing import Dict, Tuple, Sequence, Union
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import difflib
from scipy.optimize import curve_fit
import scipy as sp
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import convolve2d
from scipy.stats import norm
#from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error
from scipy.special import factorial

def get_overlap(s1, s2):
    """Longest contiguous matching subsequence
    s1 = "CGATTCCAGGCTCCCCACGGGGTACCCATAACTTGACAGTAGATCTC"
    s2 = "GGCTCCCCACGGGGTACCCATAACTTGACAGTAGATCTCGTCCAGACCCCTAGC"
    returns = GGCTCCCCACGGGGTACCCATAACTTGACAGTAGATCTC
    """
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]


def swarmplot_x(y, nbins=None, spread = 0.2):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    spread = spread
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 2

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = spread / (nmax // spread)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (1 + j / 3 + np.arange(len(b))) * dx
            x[b] = (1 + j / 3 + np.arange(len(b))) * -dx

    return x

# Define the sum of Gaussian function
def sum_of_gaussians_ChatGPT(x, *params):
    n_gaussians = len(params) // 3
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(n_gaussians):
        mean = params[i]
        amplitude = params[n_gaussians + i]
        sigma = params[2 * n_gaussians + i]
        y += amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
    return y

# Function to fit the model
def fit_sum_of_gaussians(mean, amplitude, sigma, number_of_gaussian, x_interpolated, y_interpolated):
    initial_guess = [mean, amplitude, sigma]* int(number_of_gaussian)
    try:
        popt, pcov = curve_fit(sum_of_gaussians_ChatGPT, x_interpolated, y_interpolated, p0=initial_guess, maxfev=100000)
        residuals = y_interpolated - sum_of_gaussians_ChatGPT(x_interpolated, *popt)
        fit_score = np.sum(residuals ** 2)
        return fit_score, initial_guess
    except RuntimeError:
        pass

def generate_ticks(input_array, num_ticks=5, round_to=0.05):
    min_val = np.min(input_array)
    max_val = np.max(input_array)

    # Round min_val down and max_val up to nearest multiple of round_to
    min_val = np.floor(min_val / round_to) * round_to
    max_val = np.ceil(max_val / round_to) * round_to

    ticks = np.linspace(min_val, max_val, num_ticks)

    return ticks


 
def find_spatial_STA_center(reM, checkerboard, cluster, window_size=1, pixels_to_mirror=1, interpolation_value=40, significance_level=0.00000001,
                            n_pixel_x_border = 5, n_pixel_y_border = 3):
   
    """
    Finds the spatial center of a Spike Triggered Average (STA) and performs various analyses to classify and process the STA.
    
    The function first identifies the spatial center of the STA and classifies it as "outside," "border," or "inside." It then processes
    the STA frames by applying Gaussian filtering, interpolation, and convolution operations. The processed STA is analyzed to determine
    its statistical significance, and various output metrics are provided for further analysis.

    Parameters:
    ----------
    reM : The record master array containing spike and stimulus data.
    checkerboard : The checkerboard stimulus data, as computed by the "process_sta_batch" function.
    cluster : int
        The index of the cluster for which the STA is being computed.
    window_size : int, optional, default=1
        The size of the window for the convolution operation. A value of 1 is found to work best in most cases.
    pixels_to_mirror : int, optional, default=1
        The number of pixels to mirror on each side of the frame to avoid border effects.
    interpolation_value : int, optional, default=40
        The factor by which to interpolate the STA frames, increasing their resolution.
    significance_level : float, optional, default=0.00000001
        The threshold for statistical significance in the Z-score calculation.
    n_pixel_x_border : int, optional, default=5
        The number of pixels from the edge to consider as the border in the x-direction.
    n_pixel_y_border : int, optional, default=3
        The number of pixels from the edge to consider as the border in the y-direction.

    Returns:
    -------
    spatial_STA_center_dict : dict
        A dictionary containing various metrics and data related to the spatial STA analysis.

        'checkerboard' : ndarray
            The raw STA data, a 3D array of shape (16, 32, 18), representing 16 frames of a 32x18 pixel grid.
        's-STA_center_classification' : str
            The classification of the STA center, which can be "outside," "border," or "inside."
        's-STA_frame_with_max_value' : ndarray
            A single frame (32x18) from the raw STA where the pixel values reached their maximum.
        's-STA_center_coord' : tuple of float
            The (x, y) coordinates of the STA center in the 32x18 pixel grid.
        's-STA_frame_with_max_value_processed' : ndarray
            The frame after Gaussian filtering, interpolation, and removal of mirrored pixels.
        's-STA_frame_with_max_value_processed_center_coord' : tuple of float
            The "new" (x, y) coordinates of the STA center after processing.
        's-STA_significance_bool' : bool
            Indicates whether the STA center is statistically significant.
        's-STA_significance_value' : float
            The p-value associated with the STA center's significance.
        's-STA_center_intensities_value_normalized' : list of float
            The normalized values of pixel intensities over the 16 frames measured at the center of the receptive filed
        's-STA_processed' : ndarray
            a 3D array of shape (16, 720, 1280) containing the 16 processed frames after Gaussian filtering, interpolation, 
            and removal of mirrored pixels.
    """
    
    # Retrieve the mapping of cell clusters
    map_dict  = reM[0]["S_matrix"].attrs["cell_map"]
    
    # Find the index of the cluster in the reM dataset
    cluster_index = map_dict[str(cluster)]
    
    # Extract the data for the given cluster from the checkerboard dataset
    raw_data = checkerboard[0][cluster_index]
    time = int(raw_data.shape[0])
    width = int(raw_data.shape[1])
    height = int(raw_data.shape[2])
    max_val_list = []
    
    # Iterate through all time points and apply Gaussian filter to each frame
    for time_point in range(time):
        single_frame = abs(sp.ndimage.gaussian_filter(raw_data[time_point,:,:], sigma=2))
        max_val_list.append(np.max(single_frame))

    # Find the frame with the maximum value
    frame_with_max_value = raw_data[np.argmax(max_val_list),:,:]

    # Normalize the frame values to the range 0-255 and convert to uint8
    frame_with_max_value = ((frame_with_max_value - frame_with_max_value.min()) / (frame_with_max_value.max()-frame_with_max_value.min())) * 255
    frame_with_max_value = frame_with_max_value.astype("uint8")

    # Mirror the pixels at the border of the frame to avoid border effects
    mirrored_frame = np.zeros((frame_with_max_value.shape[0] + 2 * pixels_to_mirror, frame_with_max_value.shape[1] + 2 * pixels_to_mirror))
    mirrored_frame[pixels_to_mirror:-pixels_to_mirror, pixels_to_mirror:-pixels_to_mirror] = frame_with_max_value
    mirrored_frame[pixels_to_mirror:-pixels_to_mirror, :pixels_to_mirror] = frame_with_max_value[:, :pixels_to_mirror][:, ::-1]
    mirrored_frame[pixels_to_mirror:-pixels_to_mirror, -pixels_to_mirror:] = frame_with_max_value[:, -pixels_to_mirror:][:, ::-1]
    mirrored_frame[:pixels_to_mirror, pixels_to_mirror:-pixels_to_mirror] = frame_with_max_value[:pixels_to_mirror, :][::-1, :]
    mirrored_frame[-pixels_to_mirror:, pixels_to_mirror:-pixels_to_mirror] = frame_with_max_value[-pixels_to_mirror:, :][::-1, :]
    mirrored_frame[:pixels_to_mirror, :pixels_to_mirror] = mirrored_frame[:pixels_to_mirror, pixels_to_mirror:2*pixels_to_mirror][:, ::-1]
    mirrored_frame[:pixels_to_mirror, -pixels_to_mirror:] = mirrored_frame[:pixels_to_mirror, -2*pixels_to_mirror:-pixels_to_mirror][:, ::-1]
    mirrored_frame[-pixels_to_mirror:, :pixels_to_mirror] = mirrored_frame[-2*pixels_to_mirror:-pixels_to_mirror, :pixels_to_mirror][:, ::-1]
    mirrored_frame[-pixels_to_mirror:, -pixels_to_mirror:] = mirrored_frame[-2*pixels_to_mirror:-pixels_to_mirror, -pixels_to_mirror:][:, ::-1]
    
    # Interpolate the frame to increase the resolution
    interpolated_frame = zoom(mirrored_frame, interpolation_value)

    # Apply a Gaussian filter to smooth the frame
    gaussian_filtered_frame = gaussian_filter(interpolated_frame, sigma=20)

    # Convolve the frame with a kernel to get the sum over the window, which helps to find the significant spot over a region instead of per pixel
    kernel = np.ones((window_size,window_size))
    convolved = convolve2d(gaussian_filtered_frame, kernel, mode='valid')

    # Find the maximum and minimum values and their coordinates in the convolved frame
    max_value = convolved.max()
    max_coord = np.unravel_index(convolved.argmax(), convolved.shape)
    min_coord = np.unravel_index(convolved.argmin(), convolved.shape)

    # Calculate the Z-score for the maximum and minimum values to find the most significant spot
    mean_value = np.mean(convolved)
    std_value = np.std(convolved)
    max_value = gaussian_filtered_frame[max_coord]
    min_value = gaussian_filtered_frame[min_coord]
    z_score_max = (max_value - mean_value) / std_value
    z_score_min = (min_value - mean_value) / std_value

    # Calculate the p-value for the maximum and minimum Z-scores
    p_val_max = 1 - norm.cdf(z_score_max)
    p_val_min = norm.cdf(z_score_min)
    pixels_to_remove = interpolation_value * pixels_to_mirror

    # Determine the most significant coordinate
    coord = max_coord if p_val_max < p_val_min else min_coord 
    coord = coord[::-1]

    # Determine whether the most significant spot is statistically significant
    significant = min(p_val_max, p_val_min) < significance_level
    significance = min(p_val_max, p_val_min)

    # Calculate the new center after processing
    new_center = (coord[0] - pixels_to_remove, coord[1] - pixels_to_remove)

    # Remove the mirrored pixels from the processed frames
    processed_frame = gaussian_filtered_frame[pixels_to_remove:-pixels_to_remove, pixels_to_remove:-pixels_to_remove]

    # Calculate the STA center in the spatial domain
    spatial_STA_center = (new_center[0] / interpolation_value, new_center[1] / interpolation_value)

 # Determine the classification of the STA center
    y_max, x_max = frame_with_max_value.shape
    x, y = spatial_STA_center
    if significant == False:
        classification = "outside"
    
        entire_spatial_STA_processed = None
        normalized_spatial_STA = None
    
    else:
        # Extract dimensions directly from the processed_frame
        processed_height = processed_frame.shape[0]
        processed_width = processed_frame.shape[1]

        # Create the empty 3D array with the correct dimensions
        entire_spatial_STA_processed = np.zeros((time, processed_height, processed_width))
        
        # Empty list containing the mean values of the sta-center
        spatial_STA_center_values = []
        
        # Process each frame
        for time_point in range(time):
            # Iterate through all time points and apply Gaussian filter to each frame
            #single_frame = sp.ndimage.gaussian_filter(raw_data[time_point,:,:], sigma=2)
            single_frame = raw_data[time_point,:,:]
            
            # Normalize the frame values to the range 0-255 and convert to uint8
            single_frame_normalized = ((single_frame - raw_data.min()) / (raw_data.max()-raw_data.min())) * 255
            single_frame_unit8 = single_frame_normalized.astype("uint8")
            
            # Mirror the pixels at the border of the frame to avoid border effects
            mirrored_frame = np.zeros((single_frame_unit8.shape[0] + 2 * pixels_to_mirror, single_frame_unit8.shape[1] + 2 * pixels_to_mirror))
            mirrored_frame[pixels_to_mirror:-pixels_to_mirror, pixels_to_mirror:-pixels_to_mirror] = single_frame_unit8
            mirrored_frame[pixels_to_mirror:-pixels_to_mirror, :pixels_to_mirror] = single_frame_unit8[:, :pixels_to_mirror][:, ::-1]
            mirrored_frame[pixels_to_mirror:-pixels_to_mirror, -pixels_to_mirror:] = single_frame_unit8[:, -pixels_to_mirror:][:, ::-1]
            mirrored_frame[:pixels_to_mirror, pixels_to_mirror:-pixels_to_mirror] = single_frame_unit8[:pixels_to_mirror, :][::-1, :]
            mirrored_frame[-pixels_to_mirror:, pixels_to_mirror:-pixels_to_mirror] = single_frame_unit8[-pixels_to_mirror:, :][::-1, :]
            mirrored_frame[:pixels_to_mirror, :pixels_to_mirror] = mirrored_frame[:pixels_to_mirror, pixels_to_mirror:2*pixels_to_mirror][:, ::-1]
            mirrored_frame[:pixels_to_mirror, -pixels_to_mirror:] = mirrored_frame[:pixels_to_mirror, -2*pixels_to_mirror:-pixels_to_mirror][:, ::-1]
            mirrored_frame[-pixels_to_mirror:, :pixels_to_mirror] = mirrored_frame[-2*pixels_to_mirror:-pixels_to_mirror, :pixels_to_mirror][:, ::-1]
            mirrored_frame[-pixels_to_mirror:, -pixels_to_mirror:] = mirrored_frame[-2*pixels_to_mirror:-pixels_to_mirror, -pixels_to_mirror:][:, ::-1]
            
            # Interpolate the frame to increase the resolution
            interpolated_frame = zoom(mirrored_frame, interpolation_value)
            
            # Apply a Gaussian filter to smooth the frame
            gaussian_filtered_frame_all= gaussian_filter(interpolated_frame, sigma=20)
            
            # Remove the mirrored pixels from the processed frames
            processed_frame_all = gaussian_filtered_frame_all[pixels_to_remove:-pixels_to_remove, pixels_to_remove:-pixels_to_remove]
            
            #fig, ax = plt.subplots()
            #ax.imshow(processed_frame_all, cmap='gray', aspect="auto", vmin=0, vmax=255)
            #ax.plot(new_center[0], new_center[1], 'rx')
            #circ3 = Circle(new_center, 40, fill=False, color="red", linewidth=2)
            #ax.add_patch(circ3)

            # Create an empty binary mask with the same shape as your image
            mask = np.zeros_like(processed_frame_all, dtype=bool)
            
            # Get the indices of the grid
            y_indices, x_indices = np.ogrid[:processed_frame_all.shape[0], :processed_frame_all.shape[1]]

            # Compute the distance from the center of the circle for each pixel
            distance_from_center = np.sqrt((x_indices - new_center[0])**2 + (y_indices - new_center[1])**2)

            # Set pixels inside the circle in the mask to 1 (True)
            mask[distance_from_center <= 40] = 1
            
            #plt.figure()
            #plt.hist(inside_values, bins=50, alpha=0.5, label='Inside Circle', density=True)
            #plt.hist(outside_values, bins=50, alpha=0.5, label='Outside Circle', density=True)
            #plt.legend(loc='upper right')
            #plt.show()
            
            mean_value_sta_center = processed_frame_all[mask].mean()
            spatial_STA_center_values.append(mean_value_sta_center)
            
            # Append the processed frame to the empty 3D array
            entire_spatial_STA_processed[time_point, :, :] = processed_frame_all
            
        # Calculate the mean
        mean_value = np.mean(spatial_STA_center_values)
        
        # Find the value that is most distant from the mean in absolute terms
        most_distant_value = max(spatial_STA_center_values, key=lambda x: abs(x - mean_value))
        
        # Determine if this most distant value is greater than or less than the mean
        if most_distant_value > mean_value:
            normalization_factor = most_distant_value - mean_value
        else:
            normalization_factor = mean_value - most_distant_value
            
        # Normalize the rest of the values
        normalized_spatial_STA = [(x - mean_value) / normalization_factor for x in spatial_STA_center_values] 

        if (x < n_pixel_x_border or x > x_max - n_pixel_x_border) or (y < n_pixel_y_border or y > y_max - n_pixel_y_border):
            classification = "border"
        else:
            classification = "center"
        '''
        Check the content of normalized_spatial_STA: 
        If normalized_spatia_STA is a list of NaN this means that the center and the mask around it are completely outside
        of the frame. This happens because of the mirroring of the pixels at the border performed before. 
        In this case we have to change 'classification',  'significant', 'entire_spatial_STA_processed' and 'normalized_spatial_STA'
        ''' 
        if np.isnan(normalized_spatial_STA).any():
            classification = "outside"
            significant = False
            entire_spatial_STA_processed = None
            normalized_spatial_STA = None
            
    # Prepare the output dictionary
    spatial_STA_center_dict={
    'checkerboard': raw_data,
    's-STA_center_classification': classification,
    's-STA_frame_with_max_value': frame_with_max_value,
    's-STA_center_coord': spatial_STA_center,
    's-STA_frame_with_max_value_processed': processed_frame,
    's-STA_frame_with_max_value_processed_center_coord': new_center,
    's-STA_significance_bool': significant,
    's-STA_significance_value': significance,
    's-STA_center_intensities_value_normalized':normalized_spatial_STA,
    's-STA_processed': entire_spatial_STA_processed
    }
    
    return spatial_STA_center_dict

    #test this function:
    #reM = import_record("/Users/matteotripodi/Desktop/KilosortreM/reM_M027_paired(M028)_KILOSORT_2022-05-20_awake_untreated_day.h5")
    #map_dict  = reM[0]["S_matrix"].attrs["cell_map"]
    #clusters_to_include = [0,2,3,11,12,13,20,21,28,29,33,55,57,58,62,63]
    # pipe = Data_Pipe(reM, ["checkerboard", "S_matrix"], ["stim_inten", "spike_counts"])
    # pipe += "checkerboard"
    # response_dict = {}
    # response_dict["checkerboard"] = process_sta_batch(**pipe[0], Hw=16, Fw=0, return_pval=True)
    # checkerboard = response_dict.get("checkerboard")
    # for cluster in clusters_to_include:
    #     spatial_STA_center_dict = find_spatial_STA_center(cluster=cluster)


def exponentiated_sine_wave_model(t, A, B, q, h, phi):
    '''
    Calculates the response of a neuron to a stimulus over time based on a exponentiated sine wave model.
    
    Parameters:
    - t (array-like): An array of time points at which the neuron's response is calculated.
    - A (float): The amplitude of the evoked response. Represents the maximum deviation of the response
      from the baseline (B).
    - B (float): The baseline activity level of the neuron. This is the response level in the absence
      of any stimulus or when the stimulus has no effect.
    - q (float): The exponent applied to the sine function. This parameter controls the sharpness of the
      response peaks. A higher value results in sharper peaks.
    - h (float): The frequency of the stimulus in Hz. This should match the frequency of the actual
      stimulus applied to the neuron.
    - phi (float): The phase shift of the response in radians. This parameter allows the model to account
      for delays or advances in the neuron's response relative to the stimulus onset.
      
    Returns:
    - response (array-like): An array representing the neuron's response at each time point in `t`.
      
    Example usage:
    duration = len(data) / frame_rate
    t = np.linspace(0, duration, num=len(data), endpoint=False)
    response = response_model(t=t, A=1.0, B=0.5, q=2, h=15, phi=np.pi/4)
    
    This would calculate the neuron's response over 1 second (with 500 time points) to a 15 Hz stimulus,
    with an amplitude of 1.0, a baseline activity of 0.5, a response exponent of 2, and a phase shift of pi/4 radians.

    '''
    return A * np.sin(np.pi * h * t - phi / 2)**q + B

def evaluate_fit_of_exponentiated_sine_wave_model(data, amp, B, exp, h, phi, frame_rate):
    '''
    Evaluates the fit of an exponentiated sine wave model to a given dataset by calculating the
    mean squared error (MSE) between the model's predictions and the actual data.
    
    This function generates model predictions using the specified parameters for the exponentiated
    sine wave model, then computes the MSE as a measure of fit quality. Lower MSE values indicate
    a better fit to the data.
    
    Parameters:
    - data (array-like): The actual response data of the neuron, observed over time.
    - amp (float): The amplitude of the evoked response by the model.
    - B (float): The baseline activity level of the neuron as predicted by the model.
    - exp (float): The exponent applied to the sine wave, modifying the response's sharpness.
    - h (float): The frequency of the stimulus in Hz, as used in the model.
    - phi (float): The phase shift of the model's response in radians.
    - frame_rate (float): The sampling rate at which data was recorded, in frames per second.
    
    Returns:
    - error (float): The mean squared error between the model's predictions and the actual data.
                     This value quantifies the discrepancy, with lower values indicating a better fit.
    - (amp, phi, exp) (tuple): A tuple containing the amplitude, phase shift, and exponent parameters
                               used in the model. This output is useful for identifying the parameter
                               set being evaluated.
    
    Example usage:
    error, params = evaluate_fit_of_exponentiated_sine_wave_model(data=my_neuron_data, amp=1.0, B=0.5, exp=2, h=15, phi=np.pi/4, frame_rate=60)
    
    This evaluates the fit of the model to `my_neuron_data` using a 15 Hz stimulus frequency, with
    specified amplitude, baseline, exponent, and phase shift values, at a frame rate of 60 Hz.
    '''
    
    # Calculate the duration of the data recording based on its length and frame rate
    duration = len(data) / frame_rate
    
    # Generate time points corresponding to each data point
    t = np.linspace(0, duration, num=len(data), endpoint=False)
    
    # Predict the neuron's response using the exponentiated sine wave model
    y_pred = exponentiated_sine_wave_model(t=t, A=amp, B=B, q=exp, h=h, phi=phi)
    
    # Compute the mean squared error between the predicted responses and the actual data
    error = mean_squared_error(data, y_pred)
    
    return error, (amp, phi, exp)

def doeModelInt(tfs, base, excGain, excTc, inhGain, inhTc):
    """
    Simplified difference-of-exponentials model focusing only on frequency sensitivity.
    """
    excAMP = excGain * np.exp(-excTc * tfs)
    inhAMP = inhGain * np.exp(-inhTc * tfs)
    return base + excAMP - inhAMP

def getTFFitLsq(tfs, obsresp, initparams, interp_x_axis):
    lb = [0, 0, 0, 0, 0]  # Lower bounds
    ub = [1000, np.inf, 10, np.inf, 10]  # Upper bounds
    
    # Pre-check and adjust initial parameters to ensure they are within bounds
    for i, param in enumerate(initparams):
        if param < lb[i]:
            #print(f"Initial parameter {param} at position {i} is below the lower bound {lb[i]}, adjusting...")
            initparams[i] = lb[i]
        elif param > ub[i] and ub[i] != np.inf:  # np.inf is a valid upper bound meaning no real upper limit
            #print(f"Initial parameter {param} at position {i} is above the upper bound {ub[i]}, adjusting...")
            initparams[i] = ub[i]

    # Fit the model
    popt, pcov = curve_fit(doeModelInt, tfs, obsresp, p0=initparams, bounds=(lb, ub), method='trf', maxfev=10000000)
    
    # Calculate the predicted response
    predresp = doeModelInt(tfs, *popt)
    
    # Calculate PVE
    pve = 100 * (1 - np.mean((predresp - obsresp) ** 2) / np.mean((np.mean(obsresp) - obsresp) ** 2))
    
    # Calculate the log likelihood
    log_L = -np.sum(np.log((predresp ** obsresp) * np.exp(-predresp) / factorial(np.round(obsresp).astype(int))))
    
    # Calculate interpolated response
    interpresp = doeModelInt(interp_x_axis, *popt)
    
    # Calculate likelihood bounds for comparison
    obs_mean = np.mean(obsresp)
    log_ub = -np.sum(np.log(((obsresp ** obsresp) * np.exp(-obsresp)) / factorial(np.round(obsresp).astype(int))))
    log_lb = -np.sum(np.log(((obs_mean ** obsresp) * np.exp(-obs_mean)) / factorial(np.round(obsresp).astype(int))))
    loglikelihoods = [log_L, log_lb, log_ub]
    
    return popt, loglikelihoods, pve, interpresp