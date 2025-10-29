import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class AlignResult:
    T: np.ndarray           # 2x3 affine (similarity) for B->A
    inliers: int
    total_matches: int
    success: bool

def orb_similarity_transform(A_gray, B_gray, max_features=2000, ratio=0.75, ransac_thresh=2.0):
    orb = cv2.ORB_create(nfeatures=max_features)
    kpa, desca = orb.detectAndCompute(A_gray, None)
    kpb, descb = orb.detectAndCompute(B_gray, None)
    if desca is None or descb is None or len(kpa)<6 or len(kpb)<6:
        return AlignResult(T=None, inliers=0, total_matches=0, success=False)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desca, descb, k=2)
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    if len(good) < 6:
        return AlignResult(T=None, inliers=0, total_matches=len(matches), success=False)

    src = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    T, inliers = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if T is None:
        return AlignResult(T=None, inliers=0, total_matches=len(good), success=False)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return AlignResult(T=T, inliers=inlier_count, total_matches=len(good), success=True)

def ecc_refine(A_gray, B_gray_warped, init_T, number_of_iterations=50, termination_eps=1e-5):
    # ECC works on single-channel images and returns a warp that aligns B to A.
    # We limit to Euclidean (rotation+translation) for stability.
    try:
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        cc, W = cv2.findTransformECC(A_gray, B_gray_warped, warp_matrix, warp_mode, criteria, None, 1)
        # Combine: first apply init_T, then ECC refinement
        # Equivalent single affine: W @ init_T (approx as both are 2x3; expand to 3x3)
        def to33(M):
            T = np.eye(3, dtype=np.float32)
            T[:2,:] = M
            return T
        M = to33(W) @ to33(init_T)
        out = M[:2,:]
        return out
    except cv2.error:
        return init_T

def largest_change_component(A, B, thresh=25, kernel=3):
    diff = cv2.absdiff(A, B)
    if len(diff.shape) == 3:
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        gray = diff
    # threshold
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    # morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    # components
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return None, None, None  # nothing found
    # pick largest (skip label 0 == background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    x,y,w,h,area = stats[idx]
    cx, cy = centroids[idx]
    bbox = (int(x), int(y), int(w), int(h))
    centroid = (float(cx), float(cy))
    return m, bbox, centroid

def build_centering_transform(img_shape, centroid, bbox, target_fill=0.55, rotation_deg=0.0):
    h, w = img_shape[:2]
    cx, cy = centroid
    bx, by, bw, bh = bbox
    # scale to fit bbox to target fraction of min(w,h)
    target = target_fill * min(w, h)
    scale = target / max(bw, bh) if max(bw, bh) > 0 else 1.0
    # rotation
    theta = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    # Build transform: rotate around centroid, then scale, then translate to center
    # Start from identity
    T = np.eye(3, dtype=np.float32)
    # translate centroid to origin
    T1 = np.eye(3, dtype=np.float32); T1[0,2] = -cx; T1[1,2] = -cy
    # rotation
    R33 = np.eye(3, dtype=np.float32); R33[:2,:2] = R
    # scale
    S = np.eye(3, dtype=np.float32); S[0,0] = scale; S[1,1] = scale
    # translate to image center
    T2 = np.eye(3, dtype=np.float32); T2[0,2] = w/2; T2[1,2] = h/2
    M = T2 @ S @ R33 @ T1
    return M[:2,:]  # return 2x3

def compose_magenta(before, after, thresh=25):
    diff = cv2.absdiff(after, before)  # highlight what changed going from before->after
    if len(diff.shape)==3:
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        gray = diff
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    out = before.copy()
    m = mask>0
    out[m] = (0.7*out[m] + 0.3*np.array([255,0,255])).astype(np.uint8)
    return out, mask
