import gym
import random
import numpy as np
import GPUtil
import ray
from ray import tune

from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
# import ray.rllib.agents.ppo as ppo

class SingleAgentEnv(gym.Env):
    def __init__(self, sys_params, rl_params): # env_config
        super(SingleAgentEnv, self).__init__()
        self.sys_params = sys_params
        self.rl_params = rl_params
        
        p_len = self.sys_params['p_len']
        
        self.iSimStep = 0 # initialization
        self.phase_res = torch.zeros(p_len,p_len)
        self.phase_cor = torch.zeros(p_len,p_len)
        
        self.state = 0
        
        self.observation_space = Box(low=self.rl_params['l_b']*np.array(np.ones(1)), high=self.rl_params['u_b']*np.array(np.ones(1)), dtype=float)
        self.action_space = Discrete(len(self.rl_params['action_set']))
        
        self.ad_acc_valid = 0
        self.vf_sum = 0
        self.vf_sum_total = []
        self.vf_sum_sign = 0 # vector field에 대한 부호를 저장하기 위한 변수로, (vf_sum_total[1] - vf_sum_true)에 대한 부호가 바뀌면 1로 표시
        
        self.state_total = []
        self.action_total = []
        self.reward_total = []
        
        self.Q_initial = 0
        self.Q_value = 0
        self.Q_value_total = []
        self.exp_weight = self.rl_params['exp_weight']
        
        # self.vf_sum_estimated = []
        # self.state_total = []
    
    def optical_system(self, action):
        #========================================== parameters setting ===========================================#
        
        p_len = self.sys_params['p_len']
        f_len = self.sys_params['f_len']
        
        nx = self.sys_params['nx']
        dx = self.sys_params['dx']
        n_idx = self.sys_params['n_idx']
        idx2 = self.sys_params['idx2']
        zd_list = self.sys_params['zd_list']
        
        obs_range_min = self.sys_params['obs_range_min']
        obs_range_max = self.sys_params['obs_range_max']
        vf_range_diff = self.sys_params['vf_range_diff']
        
        ad_acc = self.sys_params['ad_acc']
        Zs = self.sys_params['Zs']
        fObj = self.sys_params['fObj']
        pupil = self.sys_params['pupil']
        feature_intensity_grad_zero = self.sys_params['feature_intensity_grad_zero']
        
        device = self.rl_params['device']
        
        #==========================================================================================================#
        
        
        
        
        #======================================== tensor initialization ===========================================#
        
        v_im = torch.zeros(p_len,p_len,len(zd_list))
        fv_im = torch.zeros(p_len,p_len,len(zd_list), dtype=torch.complex128)
        fv_im_grad = torch.zeros(p_len,p_len,len(zd_list), nx, dtype=torch.complex128)
        
        Imag = torch.zeros(p_len,p_len,len(zd_list))
        fImag = torch.zeros(p_len,p_len,len(zd_list), dtype=torch.complex128)
                
        #==========================================================================================================#
        
        
        
        # corrected aberration by RL action
        self.phase_cor = torch.sum(action * Zs[3:nx+3,:,:], 0)
        
        # residual aberrations (input)    
        self.phase_res -= self.phase_cor
        scrn = (self.phase_res).to(device)
        
        
        ## Original PSF generation
        for k in range(len(zd_list)):
            kW = (zd_list[k] * Zs[idx2,:,:]).to(device) # known defocus by phase diversity
                
            P0_defocus = pupil * torch.exp(1j*(scrn + kW)) # generalized pupil function
            I0_defocus = fft.fftshift(fft.fft2(fft.ifftshift(P0_defocus))) * (dx ** 2) ## transfer function
            v_im[:,:,k] = torch.abs(I0_defocus) ** 2 # PSF
            
            P1_defocus = P0_defocus[:,:,None] * Zs[3:nx+3,:,:].permute(1,2,0).to(device)
            I1_defocus = fft.fftshift(fft.fftn(fft.ifftshift(P1_defocus), dim=(0,1))) * (dx ** 2)
            fv_im_grad[:,:,k,:] = fft.fftshift(fft.fftn(fft.ifftshift( -2*torch.imag(I1_defocus * torch.conj(I0_defocus)[:,:,None]) ), dim=(0,1))) # [range_min:range_max, range_min:range_max,:]
            
            
            ## Obtain FFT of Image intensity distribution (by inverse FFT)
            fv_im[:,:,k] = fft.fftshift(fft.fft2(fft.ifftshift(v_im[:,:,k]))) # FFT of PSF
            fImag[:,:,k] = fObj * fv_im[:,:,k] # FFT of image intensity distribution
            Imag[:,:,k] = ( fft.ifftshift(fft.ifft2(fft.fftshift(fImag[:,:,k]))) ).type(torch.float64)  # image intensity distribution
        
        ## feature image, intensity & gradient generation
        feature_image = ( (fImag[:,:,1] / fImag[:,:,0])[obs_range_min:obs_range_max,obs_range_min:obs_range_max] ).clone().detach().type(torch.complex128)
        feature_intensity = ( fft.ifftshift(fft.ifft2(fft.fftshift( feature_image ))) ).clone().detach().type(torch.float64)
                
        feature_image_grad = ( feature_image_grad_cal(fv_im, fv_im_grad, nx, f_len, obs_range_min, obs_range_max) ).clone().detach().type(torch.complex128)
        feature_intensity_grad = ( fft.ifftshift(fft.ifftn(fft.fftshift( feature_image_grad ), dim=(0,1))) ).clone().detach().type(torch.float64)
        feature_intensity_grad_diff = feature_intensity_grad - feature_intensity_grad_zero[:,:,:nx]
        
        
        for k2 in range(nx):
            img = feature_intensity_grad_diff[int(f_len/2)-vf_range_diff:int(f_len/2)+vf_range_diff, int(f_len/2)-vf_range_diff:int(f_len/2)+vf_range_diff,k2] # 4 x 4 영역 추출
            # img = feature_intensity_grad_diff[obs_diff-4:obs_diff+4,obs_diff-4:obs_diff+4,k2] # 8 x 8 영역 추출
            gx, gy = np.gradient(img)
            vf_sum = np.sum(gx) + np.sum(gy)
        
        
        return vf_sum
        
        
        
    def reset(self):
        #==========================================================================================================#
        
        # self.iSimStep = random.randint(0, len(self.sys_params['ad_acc'])-1) # At k = 0 (timestep)
        # self.ad_acc_valid = 0.5 * self.sys_params['ad_acc'][self.iSimStep,2:nx+2] # we don't know this value
        
        self.iSimStep = random.randint(1, 1) # At k = 0 (timestep)
        self.ad_acc_valid = self.iSimStep * self.sys_params['ad_acc'][0,2:nx+2] # we don't know this value
        
        phase_valid_Zernike = torch.sum(self.ad_acc_valid[:,None,None] * Zs[3:nx+3,:,:], 0)
        
        #==========================================================================================================#
        
        self.phase_res = phase_valid_Zernike
        self.state = np.zeros(self.sys_params['nx']) # np.array(self.ad_acc_valid)
        self.total_iterations = self.rl_params['total_iterations']
        
        self.state_total = []
        self.action_total = []
        self.reward_total = []
        
        self.vf_sum_total = [0]
        self.vf_sum_sign = 0
        
        self.Q_initial = 0
        self.Q_value = 0
        self.Q_value_total = []
        
        return self.state
        
        
        
    def step(self, action):
        # Apply action (0, 1, ..., n-1)
        
        # state update (transition)
        self.state += self.rl_params['action_set'][action]
        self.total_iterations -= 1 # iteration reduction
        
        vf_sum = self.optical_system(self.rl_params['action_set'][action])
        self.vf_sum_total.append(vf_sum)
        
        #### Calculate reward
        # reward = -abs(vf_sum) # 1.0 - np.abs((vf_sum_total[1] - vf_sum_true) / (vf_sum_total[0] - vf_sum_true))
        # reward = 1 - abs(vf_sum / self.vf_sum_total[-1])
        reward = abs(vf_sum)
                
        # if (self.total_iterations == (self.rl_params['total_iterations'] - 1)):
        #     reward = 0.0
        # elif (np.abs(self.vf_sum_total[-1]) < np.abs(self.vf_sum_total[-2])):
        #     reward = 1.0
        # elif (np.abs(self.vf_sum_total[-1]) == np.abs(self.vf_sum_total[-2])):
        #     reward = 0.0
        # elif (np.abs(self.vf_sum_total[-1]) > np.abs(self.vf_sum_total[-2])):
        #     reward = -1.0
        
        # reward = -abs(float(self.state)) 
                
        ### accumulate
        self.state_total.append(self.state.copy())
        self.action_total.append(self.rl_params['action_set'][action])
        self.reward_total.append(reward)
        
        
        #########################################################################################################
        
        reward_sum = 0
        n_steps = self.rl_params['total_iterations'] - self.total_iterations
        
        for i in range(1, n_steps):
            reward_sum = reward_sum + self.exp_weight * (1 - self.exp_weight) ** (n_steps-i) * self.reward_total[n_steps-i]
        
        self.Q_value = (1 - self.exp_weight) ** (n_steps) * self.Q_initial + reward_sum
        self.Q_value_total.append(self.Q_value)
        
        #########################################################################################################
        
        
        # if (self.total_iterations == (self.rl_params['total_iterations'] - 1)):
        #     reward = -abs(vf_sum)
        # else:
        #     reward = 1 - abs(vf_sum / self.vf_sum_total[-2])
        
        # Set placeholder for info
        info = {}
        
        # if (self.total_iterations == (self.rl_params['total_iterations'] - 1)):
        #     pass
        # elif np.sign(self.vf_sum_total[-1]) != np.sign(self.vf_sum_total[-2]):
        #     self.vf_sum_sign = 1
        
        # 최대 반복수 도달시 or vf_sum_total의 부호가 바뀔 경우 -> 종료
        if (self.total_iterations <= 0): # or ( (self.vf_sum_sign == 1) ):
            done = True
        else:
            done = False
        
        # # vector field 차이가 ε보다 작을 경우, 종료 (early terminating) -> ε에 대한 기준 설정?
        # if (np.abs(vf_sum_total[1] - self.params['vf_sum_true']) <= 5e-5):
        #     done = True
        
        return self.state, self.Q_value, done, info
        
    def render(self):
        # Implement visualization
        pass

