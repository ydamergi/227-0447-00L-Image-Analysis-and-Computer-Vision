import numpy as np


def compute_mx_my(calib_dict):
    """
    Given a calibration dictionary, compute mx and my (in units of [px/mm]).
    
    mx -> Number of pixels per millimeter in x direction (ie width)
    my -> Number of pixels per millimeter in y direction (ie height)
    """

    # Compute mx and my
    mx = calib_dict['width'] / ( calib_dict['aperture_w'])
    my = calib_dict['height'] / ( calib_dict['aperture_h'])

    return mx, my


def estimate_f_b(calib_dict, calib_points, n_points=None):
    """
    Estimate focal lenght f and baseline b from provided calibration points.

    Note:
    In real life multiple points are useful for calibration - in case there are erroneous points.
    Here, this is not the case. It's OK to use a single point to estimate f, b.
    
    Args:
        calib_dict (dict)           ... Incomplete calibaration dictionary
        calib_points (pd.DataFrame) ... Calibration points provided with data. (Units are given in [mm])
        n_points (int)              ... Number of points used for estimation
        
    Returns:
        f   ... Focal lenght [mm]
        b   ... Baseline [mm]
    """

    mx , my = compute_mx_my(calib_dict)
    # Extract necessary parameters from calibration dictionary
    ul = calib_points.loc[0,'ul [px]']
    ur = calib_points.loc[0,'ur [px]']
    X = calib_points.loc[0,'X [mm]']
    Z = calib_points.loc[0,'Z [mm]']
    o_x = calib_dict['o_x']

    f = (ul - o_x)*Z/(X*mx)
    fx = f*mx
    b = X - Z*(ur - o_x)/fx

    return f, b