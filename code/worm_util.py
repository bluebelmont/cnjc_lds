import autograd.numpy as np
import h5py
import os
import scipy
import collections


def load_worm_data():
    filename = 'N2 on food L_2010_04_08__11_25_23___8___1_featuresN.hdf5'
    data = h5py.File(os.path.join('../data/', filename),mode='r')
    features_timeseries = data['timeseries_data']
    eigen_worms = np.array(h5py.File('../data/master_eigen_worms_N2.mat', 'r')['eigenWorms'])

    skeleton = data['coordinates']['skeletons']
    interp_skeleton_x = nan_interpolate2d(skeleton[:,:,0].T).T
    interp_skeleton_y = nan_interpolate2d(skeleton[:,:,1].T).T

    eigen_amplitudes = np.double(np.column_stack([features_timeseries['eigen_projection_{}'.format(i)] for i in range(1,8)]))
    interp_eigen_amplitudes = nan_interpolate2d(eigen_amplitudes.T).T

    reconstructed_joints = eigen_worms.dot(interp_eigen_amplitudes.T)
    # get x, y skeleton from joint angles
    posture = angles_to_skeleton(reconstructed_joints).transpose(1,0,2) #could also get this from skel directly


    fps = 1/np.nanmedian(np.diff(data['trajectories_data']['timestamp_time']))
    max_time = data['trajectories_data']['timestamp_time'][-1]
    ret = {'coordinates': np.stack((interp_skeleton_x,interp_skeleton_y), axis=-1),
            'posture': np.array(posture), 
            'eigen_projections': np.array(interp_eigen_amplitudes),
            'eigen_worms': eigen_worms,
            'mode': features_timeseries['motion_mode'],
           'fps': fps,
           'max_time_in_seconds': max_time 
        }
    data.close()
    return ret


def angles_to_skeleton(joint_angles):
    '''
    
    Takes in an angle array and integrates over the angles to get back a skeleton.
    NB: This reconstruction assumes each segment was equally spaced so that each 
    reconstructed skeleton segment has length arclength/(numAngles + 1)
    python conversion of this function: https://github.com/aexbrown/Motif_Analysis/blob/master/eigenworms/angle2skel.m
    
    Args:
        joint_angles (np.ndarray): (n_angles, T) joint angles along body over time

    Returns:
        skeleton (np.ndarray): (n_angles + 1, T, 2) x, y position for body over time

    '''
    joint_angles = np.atleast_2d(joint_angles)
    if joint_angles.shape[0] == 1:
        joint_angles = joint_angles.T
    n_angles, T = joint_angles.shape
    

    
    skelX = np.row_stack((np.zeros(T), np.cumsum(np.cos(joint_angles)/n_angles,axis=0)))
    skelY = np.row_stack((np.zeros(T), np.cumsum(np.sin(joint_angles)/n_angles,axis=0)))
    skeleton = np.stack((skelX, skelY),axis=-1)
    return np.squeeze(skeleton)

def nan_interpolate2d(X):
    """
    Linearly interpolate missing (NaN) frames
    """
    # Timestamps x worm body
    dim1, dim2 = X.shape
    # Xout = np.zeros(shape=(dim1, dim2), dtype=float)
    t = np.arange(dim2)
    for ii in range(dim1):
        nanx = np.isnan(X[ii, :])
        nnanx = ~np.isnan(X[ii, :])

        if nanx.sum() == dim2 or nnanx.sum() == dim2:
            continue

        X[ii, nanx] = np.interp(t[nanx], t[nnanx], X[ii, nnanx])
    return X

def get_fps(data):
    fps = 1/np.nanmedian(np.diff(data['trajectories_data']['timestamp_time']))
    return fps
