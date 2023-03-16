import gym
from gym.envs.registration import register as gym_register
import enum
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import time


from mujoco_base import MuJoCoBase
from utils import *

class Environment(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.robot_geom_indices = [i for i in range(2, 14, 1)]
        self.scan_resolution = 0.5
        self.scan_range = 2
        self.scan_state = np.zeros((10, 1 + int(2*self.scan_range/self.scan_resolution), 1 + int(2*self.scan_range/self.scan_resolution)), dtype=np.float32)
        self.scanner_dim = self.scan_state.shape[0] * self.scan_state.shape[1] * self.scan_state.shape[2]
        self.max_det_per_bin = 5
        self.nDynamicObs = 20
        self.nStaticObs = 13
        
        self.goal_loc = [5.0, 10.0]
        self.interRw_loc = [[2.5, 5.0], [3.75, 7.5]]
        self.rob_loc = [self.data.qpos[0], self.data.qpos[1]]
        self.prev_dist = 0
        self.step_cnt = 0
        
        '''
        State: concatenatation of scan_state, relative dist to goal, robot speed (trans + rot), robot's current euler angle with respect to world coord (so it can relate to scan state)
        All normalized to [0, 1]
        '''
        self.state = np.concatenate((self.scan_state.flatten(), np.zeros(6, dtype=np.float32)))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.env_boundaries = gym.spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32)
        self.max_dist = np.sqrt(np.power(self.env_boundaries.high[0]-self.env_boundaries.low[0], 2) + np.power(self.env_boundaries.high[1] - self.env_boundaries.low[1], 2))
        
    def reset(self):
        # Set back to the original state
        mj.mj_resetData(self.model, self.data)
        self.rob_loc = [self.data.qpos[0], self.data.qpos[1]]
        self.scan_state = np.zeros((10, 1 + int(2*self.scan_range/self.scan_resolution), 1 + int(2*self.scan_range/self.scan_resolution)), dtype=np.float32)
        self.state = np.concatenate((self.scan_state.flatten(), np.zeros(6, dtype=np.float32)))
        self.obstacle_pos_reset()

        #Relative dist to goal
        self.state[self.scanner_dim + 0] = np.sqrt(np.power(self.goal_loc[0] - self.rob_loc[0], 2) + np.power(self.goal_loc[1] - self.rob_loc[1], 2))/self.max_dist
        self.prev_dist = np.copy(self.state[self.scanner_dim + 0])
        x_speed = self.data.qvel[0]
        y_speed = self.data.qvel[1]
        self.state[self.scanner_dim + 1] = np.clip(np.sqrt(np.power(x_speed, 2) + np.power(y_speed, 2))/2, 0, 1)
        self.state[self.scanner_dim + 2] = np.clip((self.data.qvel[5]+1)/2, 0, 1)
        rx,ry,rz,rw = quaternion_from_euler(135, 0, 0)
        self.data.qpos[3] = rx
        self.data.qpos[4] = ry
        self.data.qpos[5] = rz
        self.data.qpos[6] = rw
        self.state[self.scanner_dim + 3] = 0.5 + 135/360 #starting angle = 135. Normalized between [-180, 180] to [0, 1]
        self.state[self.scanner_dim + 4] = self.rob_loc[0]/self.env_boundaries.high[0]
        self.state[self.scanner_dim + 5] = self.rob_loc[1]/self.env_boundaries.high[1]
        self.state = np.round(self.state, 4)
        self.step_cnt = 0

        return self.state
        
    def sensor(self):
        
        #Checking collision with objects
        for i in range(self.data.ncon):

            for ridx in self.robot_geom_indices:
                if (self.data.contact[i].geom1 == ridx and self.data.contact[i].geom2 >1) or (self.data.contact[i].geom2 == ridx and self.data.contact[i].geom1 > 1):
                    return True, False #collision
                
        #Checking the robot's position within the environment boundaries
        if(self.rob_loc[0]<self.env_boundaries.low[0] or self.rob_loc[0]>self.env_boundaries.high[0] or self.rob_loc[1] < self.env_boundaries.low[0] or self.rob_loc[1] > self.env_boundaries.high[0]):
            return False, True
        
        return False, False #no collision
       
    def scanner(self):
        
        '''
        Scanner is currently in world coord.
        TODO: update it to the robot coord
        '''
        
        robot_pos_x = self.data.qpos[0]
        robot_pos_y = self.data.qpos[1]
        center_idx_x = int(self.scan_state.shape[2]/2)
        center_idx_y = int(self.scan_state.shape[1]/2)
        
        for i in range(self.scan_state.shape[0]-1, 0, -1):
            self.scan_state[i] = np.copy(self.scan_state[i-1])
        self.scan_state[0] = np.zeros((1 + int(2*self.scan_range/self.scan_resolution), 1 + int(2*self.scan_range/self.scan_resolution)), dtype=np.float32)
        
        for i in range(self.nStaticObs + self.nDynamicObs):
            obs_pos_x = self.data.qpos[11+7*i+0]
            obs_pos_y = self.data.qpos[11+7*i+1]
            
            delta_x = obs_pos_x - robot_pos_x
            delta_y = obs_pos_y - robot_pos_y
            idx_x_offset = 0
            idx_y_offset = 0
            if(np.abs(delta_x)<self.scan_range and np.abs(delta_y)<self.scan_range):

                if(delta_x>0):
                    idx_x_offset = 1
                else:
                    idx_x_offset = -1
                if(delta_y>0):
                    idx_y_offset = 1
                else:
                    idx_y_offset = -1
                
                normalizedDist = np.sqrt(np.power(delta_x,2)+np.power(delta_y,2))/np.sqrt(2*np.power(self.scan_range,2))
                currVal = self.scan_state[0, center_idx_y -idx_y_offset - int(delta_y/self.scan_resolution), center_idx_x -idx_x_offset - int(delta_x/self.scan_resolution)] 
                self.scan_state[0, center_idx_y -idx_y_offset - int(delta_y/self.scan_resolution), center_idx_x -idx_x_offset - int(delta_x/self.scan_resolution)] = max(currVal, 1-normalizedDist)
          
        #self.scan_state /= self.max_det_per_bin
        
        for i in range(self.scanner_dim):
            self.state[i] = self.scan_state.flatten()[i]

    def obstacle_pos_reset(self):
        for i in range(self.nDynamicObs):
            if(i%2==0):
                self.data.qpos[11+7*i+0] = (i+2)%8 + 1
                self.data.qpos[11+7*i+1] = 10 - i
            else:
                self.data.qpos[11+7*i+0] = (i+2)%8 + 1
                self.data.qpos[11+7*i+1] = i
       
    def obstacle_control(self):
        
        for i in range(self.nDynamicObs):
            self.data.qvel[10 + 6*i + 0] = -2*((i%2)-0.5)
            self.data.qvel[10 + 6*i + 1] = -2*(-(i%2)+0.5)
            self.data.qvel[10 + 6*i + 2] = 0
            
            self.data.qvel[10 + 6*i + 3] = 0
            self.data.qvel[10 + 6*i + 4] = 0
            self.data.qvel[10 + 6*i + 5] = 0
            
            if(self.data.qpos[11 + 7*i + 0] <0 or self.data.qpos[11+7*i+0]>10):
                if(i%2):
                    self.data.qpos[11+7*i+0] = (i+2)%8 + 1
                    self.data.qpos[11+7*i+1] = 0
                else:
                    self.data.qpos[11+7*i+0] = (i+2)%8 + 1
                    self.data.qpos[11+7*i+1] = 10
            if(self.data.qpos[11 + 7*i + 1] <0 or self.data.qpos[11+7*i+1]>10):
                if(i%2):
                    self.data.qpos[11+7*i+0] = (i+2)%8 + 1 
                    self.data.qpos[11+7*i+1] = 0
                else:
                    self.data.qpos[11+7*i+0] = (i+2)%8 
                    self.data.qpos[11+7*i+1] = 10
                
    def intermediateReward(self):
        
        for inter_rw_loc in self.interRw_loc:
            curr = np.sqrt(np.power(self.rob_loc[0] - inter_rw_loc[0], 2) + np.power(self.rob_loc[1] - inter_rw_loc[1], 2) )
            if (curr <= 2.5):
                return 5*(1-curr)
        return 0

    def step(self, action):
        assert action[0]<=1 and action[0]>=-1
        assert action[1]<=1 and action[1]>=-1

        tSpeed, rSpeed = action #input range [-1, 1]
        
        #tSpeed range [0, 2]
        #rSpeed range [-1, 1]
        tSpeed = (tSpeed+1)
        apply_speed(self.data.qpos[3],self.data.qpos[4],self.data.qpos[5],self.data.qpos[6], tSpeed, rSpeed, self.data)
        
        #Also apply random motion to dynamic obstacles
        self.obstacle_control()
        
        simstart = self.data.time
        while(self.data.time - simstart < 0.01):
            mj.mj_step(self.model, self.data)
            
        self.scanner()
        
        #update current state
        self.rob_loc = [self.data.qpos[0], self.data.qpos[1]]
        self.state[self.scanner_dim + 0] = np.sqrt(np.power(self.goal_loc[0] - self.rob_loc[0], 2) + np.power(self.goal_loc[1] - self.rob_loc[1], 2))/self.max_dist
        x_speed = self.data.qvel[0]
        y_speed = self.data.qvel[1]
        self.state[self.scanner_dim + 1] = np.clip(np.sqrt(np.power(x_speed, 2) + np.power(y_speed, 2))/2, 0.0, 1.0)
        self.state[self.scanner_dim + 2] = np.clip((self.data.qvel[5]+1)/2, 0.0, 1.0)
        rx, _ , _ = euler_from_quaternion(self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6])
        self.state[self.scanner_dim + 3] = 0.5 + rx/360
        self.state[self.scanner_dim + 4] = self.rob_loc[0]/self.env_boundaries.high[0]
        self.state[self.scanner_dim + 5] = self.rob_loc[1]/self.env_boundaries.high[1]
        self.state = np.round(self.state, 4)
        self.step_cnt += 1

        bCollision, bInvalidPlace = self.sensor()
        if(bCollision):
            return self.state, -2.5, True, {}
        
        if(bInvalidPlace):
            return self.state, -10.0, True, {}
        
        if self.state[self.scanner_dim + 0]<0.01:
            #Robot is considered reached the target when relative distance to target is <0.01
            print("Reached the Goal!")
            return self.state, 100.0, True, {}

        if(self.step_cnt>2000):
            return self.state, -10.0, True, {}
        
        #The current relative distance to the goal should be smaller than the previous one
        #Rewarded by the amount of progression // penalty by retrogression
        delta_dist = self.prev_dist - self.state[self.scanner_dim + 0]
        self.prev_dist = np.copy(self.state[self.scanner_dim + 0])
    
        #Small reward to the faster tSpeed. Want to reach the target asap
        return self.state, float(1000*delta_dist), False, {}
        
        
    def render(self, agent):
        '''
        MuJoCo simulation logic for graphics visualization
        '''
        
        self.window_init()
        # Set camera configuration for MuJoCo graphics rendering
        self.cam.azimuth = 45
        self.cam.elevation = -30.588379
        self.cam.distance = 3.0
        self.cam.lookat = np.array([2.5, 2.5, 0])
        s = self.reset()
        done = False
        step_cnt = 0
        while not glfw.window_should_close(self.window):
                        
            if(done):
                s = self.reset()
                step_cnt = 0

            action, _ = agent.predict(s, deterministic=True)

            ns, r, done, _ = self.step(action)
            step_cnt+=1
            s = ns
            
            
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Show joint frames
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = 1

            # Update scene and render
            self.cam.lookat = np.array([self.data.qpos[0], self.data.qpos[1], 0])
            rx, _, _ = euler_from_quaternion(self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6])
            self.cam.azimuth = 180-rx
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            if(not done):
                mj.mjr_text(200, 'Step: '+str(step_cnt)+' - Reward: '+str(np.round(r, 3)), self.context, 0.1, 0.1, 0, 1, 0)
                

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()
        
        
