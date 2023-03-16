import math
import numpy as np
import gym
import torch

ENV_ID = "navigator"

def euler_from_quaternion(x, y, z, w):
        """
        Converts a quaternion into euler angles
        rx, ry, rz (counterclockwise in degrees)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        rx = 180*roll_x/np.pi
        ry = 180*pitch_y/np.pi
        rz = 180*yaw_z/np.pi
     
        return rx, ry, rz

def quaternion_from_euler(rx, ry, rz):
    """
        Converts euler angles into a quaternion
        qx, qy, qz, qw
    """
    
    roll = np.pi*rx/180
    pitch = np.pi*ry/180
    yaw = np.pi*rz/180

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
    return qx, qy, qz, qw


def apply_speed(x, y, z, w, trans_speed, rot_speed, mjc_data):
    
    mjc_data.qvel[3] = 0
    mjc_data.qvel[4] = 0
    mjc_data.qvel[5] = rot_speed
    
    rx, _ , _ = euler_from_quaternion(x, y, z, w)
    
    angle = (180 - rx)*np.pi/180
    
    x_speed = trans_speed*np.cos(angle)
    y_speed = trans_speed*np.sin(angle)
    
    mjc_data.qvel[0] = x_speed
    mjc_data.qvel[1] = y_speed
    mjc_data.qvel[2] = 0
    
    return


def make_env(xml_path):
    env = gym.make(ENV_ID, xml_path=xml_path)

    return env


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    