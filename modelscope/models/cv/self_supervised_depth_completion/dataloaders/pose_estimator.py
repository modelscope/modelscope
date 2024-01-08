import cv2
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def convert_2d_to_3d(u, v, z, K):
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    return (x, y, z)


def feature_match(img1, img2):
    r''' Find features on both images and match them pairwise
   '''
    max_n_features = 1000
    # max_n_features = 500
    use_flann = False  # better not use flann

    detector = cv2.xfeatures2d.SIFT_create(max_n_features)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if (des1 is None) or (des2 is None):
        return [], []
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    if use_flann:
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        matcher = cv2.DescriptorMatcher().create('BruteForce')
        matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def get_pose_pnp(rgb_curr, rgb_near, depth_curr, K):
    gray_curr = rgb2gray(rgb_curr).astype(np.uint8)
    gray_near = rgb2gray(rgb_near).astype(np.uint8)
    height, width = gray_curr.shape

    pts2d_curr, pts2d_near = feature_match(gray_curr,
                                           gray_near)  # feature matching

    # dilation of depth
    kernel = np.ones((4, 4), np.uint8)
    depth_curr_dilated = cv2.dilate(depth_curr, kernel)

    # extract 3d pts
    pts3d_curr = []
    pts2d_near_filtered = [
    ]  # keep only feature points with depth in the current frame
    for i, pt2d in enumerate(pts2d_curr):
        # print(pt2d)
        u, v = pt2d[0], pt2d[1]
        z = depth_curr_dilated[v, u]
        if z > 0:
            xyz_curr = convert_2d_to_3d(u, v, z, K)
            pts3d_curr.append(xyz_curr)
            pts2d_near_filtered.append(pts2d_near[i])

    # the minimal number of points accepted by solvePnP is 4:
    if len(pts3d_curr) >= 4 and len(pts2d_near_filtered) >= 4:
        pts3d_curr = np.expand_dims(np.array(pts3d_curr).astype(np.float32),
                                    axis=1)
        pts2d_near_filtered = np.expand_dims(
            np.array(pts2d_near_filtered).astype(np.float32), axis=1)

        # ransac
        ret = cv2.solvePnPRansac(pts3d_curr,
                                 pts2d_near_filtered,
                                 K,
                                 distCoeffs=None)
        success = ret[0]
        rotation_vector = ret[1]
        translation_vector = ret[2]
        return (success, rotation_vector, translation_vector)
    else:
        return (0, None, None)
