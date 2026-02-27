import numpy as np
import mujoco

def body_yaw_from_xmat(xmat_flat):
    """MuJoCo 的 xmat 是 row-major 的 9 元素，取出 yaw（绕 z 轴）"""
    M = np.array(xmat_flat).reshape(3, 3)
    # yaw = atan2(R10, R00)  (即 atan2(sin, cos))
    return float(np.arctan2(M[1, 0], M[0, 0]))

def angle_wrap_pi(a):
    """把角度差 wrap 到 [-pi, pi]"""
    return (a + np.pi) % (2*np.pi) - np.pi

def check_xy_pose_match(model, data, body_a, body_b, pos_tol=0.01, yaw_tol_deg=5.0):
    """
    检查两个 body 是否在 XY 平面位置接近且朝向（yaw）接近
    pos_tol: 允许的 XY 距离（米）
    yaw_tol_deg: 允许的 yaw 误差（度）
    返回: (is_match, xy_dist, yaw_err_deg)
    """
    ida = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_a)
    idb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_b)

    pa = data.xpos[ida][:2]
    pb = data.xpos[idb][:2]
    xy_dist = float(np.linalg.norm(pa - pb))

    yawa = body_yaw_from_xmat(data.xmat[ida])
    yawb = body_yaw_from_xmat(data.xmat[idb])
    yaw_err = angle_wrap_pi(yawa - yawb)
    yaw_err_deg = float(abs(yaw_err) * 180.0 / np.pi)

    ok = (xy_dist <= pos_tol) and (yaw_err_deg <= yaw_tol_deg)
    return ok, xy_dist, yaw_err_deg