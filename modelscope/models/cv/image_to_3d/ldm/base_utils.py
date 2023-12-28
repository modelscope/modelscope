import pickle
import numpy as np
import cv2
from skimage.io import imread


def save_pickle(data, pkl_path):
    # os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def draw_epipolar_line(F, img0, img1, pt0, color):
    h1,w1=img1.shape[:2]
    hpt = np.asarray([pt0[0], pt0[1], 1], dtype=np.float32)[:, None]
    l = F @ hpt
    l = l[:, 0]
    a, b, c = l[0], l[1], l[2]
    pt1 = np.asarray([0, -c / b]).astype(np.int32)
    pt2 = np.asarray([w1, (-a * w1 - c) / b]).astype(np.int32)

    img0 = cv2.circle(img0, tuple(pt0.astype(np.int32)), 5, color, 2)
    img1 = cv2.line(img1, tuple(pt1), tuple(pt2), color, 2)
    return img0, img1

def draw_epipolar_lines(F, img0, img1,num=20):
    img0,img1=img0.copy(),img1.copy()
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    for k in range(num):
        color = np.random.randint(0, 255, [3], dtype=np.int32)
        color = [int(c) for c in color]
        pt = np.random.uniform(0, 1, 2)
        pt[0] *= w0
        pt[1] *= h0
        pt = pt.astype(np.int32)
        img0, img1 = draw_epipolar_line(F, img0, img1, pt, color)

    return img0, img1

def compute_F(K1, K2, Rt0, Rt1=None):
    if Rt1 is None:
        R, t = Rt0[:,:3], Rt0[:,3:]
    else:
        Rt = compute_dR_dt(Rt0,Rt1)
        R, t = Rt[:,:3], Rt[:,3:]
    A = K1 @ R.T @ t # [3,1]
    C = np.asarray([[0,-A[2,0],A[1,0]],
                    [A[2,0],0,-A[0,0]],
                    [-A[1,0],A[0,0],0]])
    F = (np.linalg.inv(K2)).T @ R @ K1.T @ C
    return F

def compute_dR_dt(Rt0, Rt1):
    R0, t0 = Rt0[:,:3], Rt0[:,3:]
    R1, t1 = Rt1[:,:3], Rt1[:,3:]
    dR = np.dot(R1, R0.T)
    dt = t1 - np.dot(dR, t0)
    return np.concatenate([dR, dt], -1)

def concat_images(img0,img1,vert=False):
    if not vert:
        h0,h1=img0.shape[0],img1.shape[0],
        if h0<h1: img0=cv2.copyMakeBorder(img0,0,h1-h0,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        if h1<h0: img1=cv2.copyMakeBorder(img1,0,h0-h1,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=1)
    else:
        w0,w1=img0.shape[1],img1.shape[1]
        if w0<w1: img0=cv2.copyMakeBorder(img0,0,0,0,w1-w0,borderType=cv2.BORDER_CONSTANT,value=0)
        if w1<w0: img1=cv2.copyMakeBorder(img1,0,0,0,w0-w1,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=0)

    return img

def concat_images_list(*args,vert=False):
    if len(args)==1: return args[0]
    img_out=args[0]
    for img in args[1:]:
        img_out=concat_images(img_out,img,vert)
    return img_out


def pose_inverse(pose):
    R = pose[:,:3].T
    t = - R @ pose[:,3:]
    return np.concatenate([R,t],-1)

def project_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    mask0 = (np.abs(dpt)<1e-4) & (np.abs(dpt)>0)
    if np.sum(mask0)>0: dpt[mask0]=1e-4
    mask1=(np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1)>0: dpt[mask1]=-1e-4
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt


def draw_keypoints(img, kps, colors=None, radius=2):
    out_img=img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color=[int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, -1)
        else:
            cv2.circle(out_img, tuple(pt), radius, (0,255,0), -1)
    return out_img


def output_points(fn,pts,colors=None):
    with open(fn, 'w') as f:
        for pi, pt in enumerate(pts):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ')
            if colors is not None:
                f.write(f'{int(colors[pi,0])} {int(colors[pi,1])} {int(colors[pi,2])}')
            f.write('\n')

DEPTH_MAX, DEPTH_MIN = 2.4, 0.6
DEPTH_VALID_MAX, DEPTH_VALID_MIN = 2.37, 0.63
def read_depth_objaverse(depth_fn):
    depth = imread(depth_fn)
    depth = depth.astype(np.float32) / 65535 * (DEPTH_MAX-DEPTH_MIN) + DEPTH_MIN
    mask = (depth > DEPTH_VALID_MIN) & (depth < DEPTH_VALID_MAX)
    return depth, mask


def mask_depth_to_pts(mask,depth,K,rgb=None):
    hs,ws=np.nonzero(mask)
    depth=depth[hs,ws]
    pts=np.asarray([ws,hs,depth],np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    if rgb is not None:
        return np.dot(pts, np.linalg.inv(K).transpose()), rgb[hs,ws]
    else:
        return np.dot(pts, np.linalg.inv(K).transpose())

def transform_points_pose(pts, pose):
    R, t = pose[:, :3], pose[:, 3]
    if len(pts.shape)==1:
        return (R @ pts[:,None] + t[:,None])[:,0]
    return pts @ R.T + t[None,:]

def pose_apply(pose,pts):
    return transform_points_pose(pts, pose)

def downsample_gaussian_blur(img, ratio):
    sigma = (1 / ratio) / 3
    # ksize=np.ceil(2*sigma)
    ksize = int(np.ceil(((sigma - 0.8) / 0.3 + 1) * 2 + 1))
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT101)
    return img