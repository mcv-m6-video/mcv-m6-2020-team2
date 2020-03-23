"""
Library: https://adamspannbauer.github.io/python_video_stab/html/module_docs.html
Is also based on this implementation
"""

import imageio
import numpy as np
import cv2
from tqdm import trange


def movingAverage(curve, radius):
  window_size = 2 * radius + 1

  f = np.ones(window_size)/window_size  # Define the filter
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') # Add padding to the boundaries

  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  curve_smoothed = curve_smoothed[radius:-radius] # Remove padding

  return curve_smoothed

def smooth(trajectory, smooth_radius):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=smooth_radius)

  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame



def motion_estimation(previous_frame, current_frame):

    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(previous_frame, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Find transformation matrix
    # m = cv2.estimateAffine2D(prev_pts, curr_pts) # .estimateRigidTransform(fullAffine=True)
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts) # .estimateRigidTransform(fullAffine=False)

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    return [dx, dy, da]


def apply_camera_motion(i, frame, transforms_smooth, w, h):

    # Extract transformations from the new transformation array
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    frame_stabilized = cv2.warpAffine(frame, m, (w, h))
    frame_stabilized = fixBorder(frame_stabilized)

    frame_out = cv2.hconcat([frame, frame_stabilized])

    # If the image is too big, resize it.
    if (frame_out.shape[1] >= 1920):
        frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))

    return frame_out

def point_feature_matching(cap, smooth_radius, output_file, to_video, video_percentage=1):
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if to_video:
        out = cv2.VideoWriter(f'{output_file}/plot_feature_matching.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    else:
        out = imageio.get_writer(f'{output_file}/plot_feature_matching.gif', fps=fps)

    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    # STEP 1: ESTIMATE MOTION
    for i in trange(n_frames -2, desc="Estimating motion"):

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        transforms[i] = motion_estimation(prev_gray, curr_gray)
        prev_gray = curr_gray

    # STEP 2: COMPUTE SMOOTH TRAJECTORY

    trajectory = np.cumsum(transforms, axis=0)
    difference = smooth(trajectory, smooth_radius) - trajectory
    transforms_smooth = transforms + difference


    # STEP 3: APPLY SMOOTH CAMERA MOTION TO FRAMES
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # Reset stream to first frame
    start = 0
    end = int(n_frames*video_percentage)
    for i in trange(start, end, desc="Apply Stabilization and saving video"):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_stabilized = apply_camera_motion(i, frame, transforms_smooth, w, h)

        if to_video:
            # don't use the concat!
            out.write(frame_stabilized)
        else:
            out.append_data(frame_stabilized)

    if not to_video:
        out.close()