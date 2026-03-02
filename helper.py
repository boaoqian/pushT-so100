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

def check_xy_pose_match(model, data, site_a_name, site_b_name, pos_tol=0.01, yaw_tol_deg=5.0):
    # 1. 获取 Site 的全局 ID
    site_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_a_name)
    site_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_b_name)

    # 2. 获取 Site 的全局位置 (xpos)
    # site.xpos 已经包含了 body 的位置和 site 自身的偏移
    pa = data.site_xpos[site_a_id][:2]
    pb = data.site_xpos[site_b_id][:2]
    xy_dist = np.linalg.norm(pa - pb)

    # 3. 获取 Site 所在 Body 的旋转矩阵
    # 注意：site 本身也有旋转，但通常我们对比 body 的朝向
    body_a_id = model.site_bodyid[site_a_id]
    body_b_id = model.site_bodyid[site_b_id]
    
    mat_a = data.xmat[body_a_id].reshape(3, 3)
    mat_b = data.xmat[body_b_id].reshape(3, 3)

    # 4. 计算 Yaw 角度 (从旋转矩阵提取)
    # 假设物体绕 Z 轴旋转，使用 atan2(R[1,0], R[0,0])
    yaw_a = np.arctan2(mat_a[1, 0], mat_a[0, 0])
    yaw_b = np.arctan2(mat_b[1, 0], mat_b[0, 0])
    
    # 角度差值处理 (Wrap to [-pi, pi])
    yaw_err = (yaw_a - yaw_b + np.pi) % (2 * np.pi) - np.pi
    yaw_err_deg = np.abs(np.degrees(yaw_err))

    is_match = (xy_dist <= pos_tol) and (yaw_err_deg <= yaw_tol_deg)
    return is_match, xy_dist, min(180-yaw_err_deg,yaw_err_deg)