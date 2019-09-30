from random import randint
from scipy.io import loadmat
import os
import pygame
import numpy as np
import math
from statistics import median
pygame.init()

# Environment HYPERPARAMETERS FOR TUNING

rew_scale_val = -10

print("#"*30)
print("Environment Hyperparameters")
print()
print("#"*30)
#


### update on 07/28:
### convex function for reward (nonlinearity), similar to the function of IDM
### previous safety constraint should prevent this situation: leading vehicle decelerate at minimum acceleration
### leading vehicle may decelerate at HV's minimum deceleration: use this one for safety constraint
### change safety constraints: guarantee no crash even when the leading car break at its maximum deceleration
### change penalty: continuous associated
### change reward: continuous associated, similar to IDM

### update on 09/10:
### correct safety constraints: not just next step, but until the leading vehicle stops
### simulate following IDM with initialized networkq
### rectifying for IDM/CACC?


data = loadmat('veh_combined.mat')['veh_combined']
# time_index, location, distance to prior vehicle, velocity, veocity difference to prior vehicle, acceleration
### future improve: heterogeneous value for each vehicle from historical data

a_min_hv = min([item for sublist in data[:,:,5] for item in sublist]) ### leading vehicle may decelerate at HV's minimum deceleration
a_max_hv = max([item for sublist in data[:,:,5] for item in sublist])
large_gap_threshold = (median([item for sublist in data[:,:,2] for item in sublist])+max([item for sublist in data[:,:,2] for item in sublist]))/2 #can we find the threshold satisfy 90% of distribution?
v_min_hv = min([item for sublist in data[:,:,3] for item in sublist])
v_max_hv = max([item for sublist in data[:,:,3] for item in sublist])
#print(v_min_hv)
#print(v_max_hv)
#print(large_gap_threshold)
#print(a_min_hv)
#print(a_max_hv)
#min_spacing = min(data[:,2,2]) #5.3< veh_l, no need to subtract veh_l
#max_spacing = max([item for sublist in data[:,:,2] for item in sublist])
#print(min_spacing)
#print(max_spacing)
#quit()

class observation_space:
    def __init__(self, num_leading_cars, num_following_cars):
        # (Velocity, Distance, Acceleration) per vehicle + (V,A) for car
        self.size = 3 * (num_leading_cars + num_following_cars + 1)
        self.n = self.size
        self.shape = (self.size,1)

class action_space:
    def __init__(self):
        # Acceleration
        self.n = 15

# Simulator needed for CAVQL
class Simulator():
    def __init__(self, num_leading_cars = 0, num_following_cars = 0):
        print("Initializing Environment...")
        # Window Size for visualization
        self.win_width = 1200
        self.win_height = 500
        self.win = None

        self.num_leading_cars = num_leading_cars
        self.num_following_cars = num_following_cars
        self.num_vehicles = self.num_leading_cars + 1 + self.num_following_cars
        self.observation_space = observation_space(num_leading_cars, num_following_cars)
        self.action_space = action_space()

        # Past stats and current displacement
        self.current_states = []
        self.current_disps = []

        # Start marker for data
        self.t_start = 10

        # Current time step
        self.t = 0

        # Max number of time steps
        self.max_t = 300

        # Parameters for reward
        # For weighting acceleration cost
        self.LAMBDA = 1 #100
        # To prevent division by zero in acceleration avg.
        self.EPSILON = 0.0001
        # To scale reward to be smaller (Helps with learning)
        self.rew_normalize = 1
        # Keeps track of negative rewards
        self.neg_rewards = []
        # Checks if crashed
        self.is_crashed = False


        # Max and Min Vel and Acc
        self.a_max = 2.0
        self.a_min = -2.0
        self.dt = 0.1 # time interval
        self.v_max = v_max_hv #16
        self.v_min = v_min_hv #0

        # Allows printing of Vel, Acc, and Gap warnings
        self.verbose = False
        # Allows for centering of states
        self.normalize = True

        # Amount of time steps to use CACC to prior to learning
        self.warmup = 100
        # Amoun of time steps to wait before using CACC again
        self.warmup_gap = 200

        self.results = {"min_gap": 0, "max_gap": 0, "violates_a": 0, "violates_v": 0, "reward": 0, "reward_disp": 0, "reward_fuel": 0, "reward_penalty": 0}

    def get_state(self,t):
        s = []
        for car in data[t][0:self.num_vehicles]:
            v = car[3]
            x = car[1]
            a = car[5]
            s = s + [v,x,a]
        return s

    def center_state(self,s):
        new_s = s
        if self.normalize:
            cav_pos = s[(self.num_leading_cars)*3+1]
            shift = [0,cav_pos,0]*self.num_vehicles
            new_s = [a - b for a, b in zip(s,shift)]
        return new_s

    def update_state(self,s,a,human=False,controller="OURS"):
        rew = 0 # penalty reward for one step
        # for leading vehicles
        v_set_next = data[self.t_start+self.t+1][0:self.num_leading_cars][:,3].reshape((1,-1))[0]
        # d = distance between any vehicle and the controlled vehicle?
        x_set_next = data[self.t_start+self.t+1][0:self.num_leading_cars][:,1].reshape((1,-1))[0] # better to use x than d, otherwise d for leading vehicles have to be calculated
        a_set_next = data[self.t_start+self.t+1][0:self.num_leading_cars][:,5].reshape((1,-1))[0]

        s_ = [0 for i in range(len(s))] # next state
        for n in range(self.num_leading_cars): # from observation in reality, from data in simulation
            s_[n*3+0] = v_set_next[n] # t: current time index
            s_[n*3+1] = x_set_next[n]
            s_[n*3+2] = a_set_next[n]

        # for controlled CAV
        v = s[self.num_leading_cars*3+0] # Location of the controlled vehicle at last step v1
        x = s[self.num_leading_cars*3+1] # if size of s is 3*(num_leading_cars+1+num_following_cars)
        a1 = s[self.num_leading_cars*3+2]#a1
        # or def CACC
        # DRL controller
        ### change penalty
        pn_a_max = rew_scale_val
        pn_a_min = rew_scale_val
        pn_a_max_power = 2
        pn_a_min_power = 2
        a_ = a
        # penalty for violating constraint for acceleration
        if a > self.a_max: # or use rectify function
            rew = rew + pn_a_max*(a/self.a_max)**pn_a_max_power ###change penalty: continuous
            self.results["violates_a"] += 1
            #if (not human) and controller == "OURS":
                #a_ = self.a_max
            if(self.verbose):
                print("Violates Max Acceleration")
                print(a)
                #quit()
        elif a < self.a_min:
            rew = rew + pn_a_min*(a/self.a_min)**pn_a_min_power ###change penalty: continuous
            self.results["violates_a"] += 1
            #if (not human) and controller == "OURS":
                #a_ = self.a_min
            if(self.verbose):
                print("Violates Min Acceleration")
                print(a)
                #quit()
        else:
            a_ = a

        #if a_- a>0:
        #    print("compareeeeeeeeeeee11111")
        #    print(a_)
        #    print(a)
        #    quit()

        if human:
            v_ = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+0]
            v3 = self.get_state(self.t_start+self.t+2)[self.num_leading_cars*3+0]
            a_ = a
        else:
            v_ = v + a1*self.dt #current
            v3 = v_ + a_*self.dt #using a_, next

        ###change penalty: continuous
        pn_v_max = rew_scale_val
        pn_v_min = rew_scale_val
        pn_v_max_power = 2
        pn_v_min_power = 2
        # penalty for violating constraint for velocity
        if v3 > self.v_max:
            rew = rew + pn_v_max*(v3-self.v_max)**pn_v_max_power###change penalty: continuous
            self.results["violates_v"] += 1
            #if (not human) and controller == "OURS":
                #v3 = self.v_max
                #a_ = (v3-v_) / self.dt
            if(self.verbose):
                print("Violates Max Vel")
                print(a_)
                #quit()
        elif v3 < self.v_min:
            rew = rew + pn_v_min*(self.v_min-v3)**pn_v_max_power ###change penalty: continuous
            self.results["violates_v"] += 1
            #if (not human) and controller == "OURS":
                #v3 = self.v_min
                #a_ = (v3-v_) / self.dt
            if(self.verbose):
                print("Violates Min Vel")
                print(a_)
                #quit()

        x_ = x + v*self.dt + (1/2)*a1*self.dt**2 #x2
        x3 = x_ + v_*self.dt + (1/2)*a_*self.dt**2 #x3
        if human:
            x_ = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+1]
            x3 = self.get_state(self.t_start+self.t+2)[self.num_leading_cars*3+1]


        min_s = 2.0 # safe gap
        #l = 6.85
        ### change safety constraints: guarantee no crash even when the leading car break at its maximum deceleration
        xl_ = self.get_state(self.t_start+self.t+1)[(self.num_leading_cars-1)*3+1] # current location of the leading vehicle at t2 xl_2, estimated in real world?
        vl_ = self.get_state(self.t_start+self.t+1)[(self.num_leading_cars-1)*3+0]# current speed vl_2
        dt_sharp_stop = -vl_/a_min_hv #time needed for the leading car's sharp stop # discretization issue?
        xl_sharp_stop = xl_ + vl_*dt_sharp_stop + (1/2)*a_min_hv*dt_sharp_stop**2 # location of leading car when after the sharp stop
        x_sharp_stop = x3 + v3*(dt_sharp_stop-self.dt) + (1/2)*self.a_min*(dt_sharp_stop-self.dt)**2  # location of controlled car when the leading car stops
        gap = xl_sharp_stop - x_sharp_stop - min_s #-l### gap after the sharp stop, long term prevention
        #print("GAP")
        #print(gap)
        #print("GAP")
        ### change penalty: continuous
        pn_safe = rew_scale_val
        pn_safe_power = 1
        # penalty for violating constraint for safety
        #print(gap)
        if gap < -1e-10:
            rew = rew + pn_safe * abs(gap) ** pn_safe_power ### change penalty: continuous
            self.results["min_gap"] += 1
            if(self.verbose):
                print("#")
                print(gap)
                print("Warning Within Gap")
                #quit()
                #print(x3)
                #print(a_)
                #print(x_)
                #print(xl_)

            #if (not human) and controller == "OURS":
                #x_sharp_stop = xl_sharp_stop - min_s #-l# to meet gap = 0
                #a_gap = (x_sharp_stop - x_ - v_*self.dt - (1/2)*self.a_min*(dt_sharp_stop-self.dt)**2) / (dt_sharp_stop*self.dt-(1/2)*self.dt**2)
                #print(a_gap)
                #print(a_)
                #a_ = a_gap
            if a_<self.a_min or a_>self.a_max:
                if(self.verbose):
                    #print(xl_3)
                    #print(x3)
                    #print(x_)
                    #print(v_)
                    #print(a_)
                    print("safety wrong acc")
                    # TODO
                    #quit()
            #if (not human) and controller == "OURS":
                #v3 = v_ + a_*self.dt
            if v3<self.v_min or v3>self.v_max:
                if(self.verbose):
                    print(v_)
                    print("safety wrong vel")
                    # TODO
                    #quit()

        ###????

        #gap = xl_3 - x3 -l - min_s
        #gap = xl_sharp_stop - x_sharp_stop - min_s #-l
        #if gap < -1e-10:
            #if(self.verbose):
                #print("Crashed")
            #print("%")
            #print(s)
            #quit()
            #self.is_crashed = True
            #rew = rew + -100000

        if gap > large_gap_threshold: ### change penalty: continuous
            rew = rew + 10000 * pn_safe * (gap-large_gap_threshold) ** pn_safe_power
            self.results["max_gap"] += 1
            #print("BADBADBAD")
            #quit()
            if(self.verbose):
                print("Warning Gap Too Large")

        s_[self.num_leading_cars*3+0] = v_
        s_[self.num_leading_cars*3+1] = x_
        s_[self.num_leading_cars*3+2] = a_
        #if a_-a != 0:
            #print("compareeeeeeeeeeee222222")
            #print(a_)
            #print(a)
            #quit()
        if human:
            s_[self.num_leading_cars*3+0] = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+0]
            s_[self.num_leading_cars*3+1] = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+1]
            s_[self.num_leading_cars*3+2] = self.get_state(self.t_start+self.t+1)[self.num_leading_cars*3+2]
        # for following num_vehicles
        #f_controllers = "HV_init" # Use controller initialized from HV to exclude IDM effects
        f_controllers = "IDM" # Designates what controller to use for following
        for n in range(self.num_leading_cars+1,self.num_leading_cars + self.num_following_cars + 1):
            v_temp = None
            x_temp = None
            a_temp = None
            if f_controllers == "IDM":
                v_temp, x_temp, a_temp = self.IDM(s,n,s_)
            elif f_controllers == "CACC":
                v_temp, x_temp, a_temp = self.CACC(s,n)
            elif f_controllers == "HV_init":
                v_temp, x_temp, a_temp = self.HV_init(s,n)
            # Add new controllers here ***
            s_[n*3+0] = v_temp
            s_[n*3+1] = x_temp
            s_[n*3+2] = a_temp
        return s,a,rew,s_,a_

    def HV_init(s,n):
        # use the initial network to generate FV's variables car by car
        return v_, x_, a_

    def IDM(self,s,n,s_):###### update by adding s_ for safety constraint
        # model specific parameters
        l = 6.85 # vehicle effective length
        s_0 = 2 # minimum gap
        T = 1.0 # time gap is different from dt

        # car following model: IDM
        # REALLY BAD!!!
        vf = s[n*3+0] # following car in this car-following pair
        xf = s[n*3+1]
        af = s[n*3+2]
        dx = s[(n-1)*3+1] - xf - l
        dv = s[(n-1)*3+0] - vf
        s_star = s_0 + max(0, vf*T + vf*dv/(2*((self.a_max*abs(self.a_min))**0.5))) #why max
        a_temp = self.a_max*(1-(vf/self.v_max)**4 -(s_star/dx)**2) + randint(0,21)*0.0001 #### update by adding random number (0,0.02) 0606
        a_temp = max(a_temp,-vf/self.dt) #

        # Update by adding constraint checking and rectifying
        ####### update from here
        # constraint for acceleration
        if a_temp > self.a_max: # or use rectify function
            a_temp = self.a_max
            if(self.verbose):#??????
                print("FC Violates Max Acceleration")
        elif a_temp < self.a_min:
            a_temp = self.a_min
            if(self.verbose):
                print("FC Violates Min Acceleration")

        v_temp =  vf + af*self.dt #v2
        v3_temp = v_temp + a_temp*self.dt #v3 using a_
        # constraint for velocity
        if v3_temp > self.v_max:
            v3_temp = self.v_max
            a_temp = (v3_temp-v_temp) / self.dt
            if(self.verbose):
                print("FC Violates Max Vel")
        elif v3_temp < self.v_min:
            v3_temp = self.v_min
            a_temp = (v3_temp-v_temp) / self.dt
            if(self.verbose):
                print("FC Violates Min Vel")

        x_temp = (vf+v_temp)/2*self.dt + xf #x2
        x3_temp = x_temp + v_temp*self.dt + (1/2)*a_temp*self.dt**2 #x3
        ### change safety constraints: guarantee no crash even when
        min_s = 2.0 # safe gap
        l = 6.85
        xl_ = s_[(n-1)*3+1] # location of the leading vehicle at t2 xl_2
        #s[(n-1)*3+1] + s[(n-1)*3+0]*self.dt + (1/2)*s[(n-1)*3+2]*self.dt**2
        #self.get_state(self.t_start+self.t+1)[(n-1)*3+1] leading vehicle is CAV, will be updated, not from data
        vl_ = s_[(n-1)*3+0] # speed of the leading vehicle at t2
        #self.get_state(self.t_start+self.t+1)[(n-1)*3+0]
        xl_3 = xl_ + vl_*self.dt + (1/2)*a_min_hv*self.dt**2 ### when the leading car break at its maximum deceleration
        gap = xl_3 - x3_temp - l - min_s ### change safety constraints: guarantee no crash even when

        #since we can not use future information in reality and current value is meaningless if collision is happening
        if gap < -1e-10:
            if(self.verbose):
                print("FC Violates safety")
            x3_temp = xl_3 - l - min_s ### change safety constraints:
            a_temp = (x3_temp - x_temp - v_temp*self.dt)*2/(self.dt**2)
            if a_temp<self.a_min or a_temp>self.a_max:
                if(self.verbose):
                    print("FC safety wrong acc")
                    #print(a_temp)
                    #input()
                    #quit()
            v_temp = vf + af*self.dt
            v3_temp = v_temp + a_temp*self.dt
            if v3_temp<self.v_min or v3_temp>self.v_max:
                if(self.verbose):
                    print("FC safety wrong vel")
                    #print(v3_temp)
                    #input()
                    #quit()
        ######## update end here
        # Add penalty for too large gap?
        ######

        return v_temp, x_temp, a_temp

    def CACC(self,s,n): # traditional controller to choose the acceleration
        l = 0#6.85
        k = 0.3
        ka = 1.0
        kv = 3.0
        kd = 0.2
        v = s[(n)*3+0]
        x = s[(n)*3+1]
        a_v = k * (self.v_max - v)
        vp = s[(n-1)*3+0]
        xp = s[(n-1)*3+1]
        ap = s[(n-1)*3+2]
        r_ref = 2
        #a_d = ka*ap + kv*(vp-v) + kd*(xp-x-l -r_ref)
        #a_ = min(a_v,a_d)
        k1 = 0.2
        k2 = 0.08
        g_d = 3
        a_d = k1*(xp-x-v*g_d)+k2*(vp-v)
        a_ = a_d
        #if a_d > 0:
            #a_ = min(a_d, a_max_hv)
        #elif a_d < 0:
            #a_ = max(a_d, a_min_hv)
        # update by adding rectify 06/13/19
        ####### update from here
        if False:
            # constraint for acceleration
            if a_ > self.a_max: # or use rectify function
                a_ = self.a_max
                if(self.verbose):#??????
                    print("CACC Violates Max Acceleration")
            elif a_ < self.a_min:
                a_ = self.a_min
                if(self.verbose):
                    print("CACC Violates Min Acceleration")

        v_ = v + s[(n)*3+2]*self.dt
        v3 = v_  + a_*self.dt

        if False:
            # constraint for velocity
            if v3 > self.v_max:
                v3 = self.v_max
                a_ = (v3-v_) / self.dt
                if(self.verbose):
                    print("CACC Violates Max Vel")
            elif v3 < self.v_min:
                v3 = self.v_min
                a_ = (v3-v_) / self.dt
                if(self.verbose):
                    print("CACC Violates Min Vel")

        x_ = (v+v_)/2*self.dt + x
        x3 = (v_+v3)/2*self.dt + x_

        if False:
            ### change safety constraints: guarantee no crash even when
            min_s = 2.0  # safe gap
            xp_ = self.get_state(self.t_start+self.t+1)[(n-1)*3+1] # location of the leading vehicle at t2 xl_2
            # extract from data since CACC replaces controller, the leading vehicle is fixed
            vp_ = self.get_state(self.t_start+self.t+1)[(n-1)*3+0]
            xp_3 = xp_ + vp_ * self.dt + (1 / 2) * a_min_hv * self.dt ** 2  ### when the leading car break at its maximum deceleration
            #l = 0
            gap = xp_3 - x3 - l - min_s  ### change safety constraints: guarantee no crash even when

            if gap < -1e-10:
                if(self.verbose):
                    print("#")
                    print(gap)
                    print("CACC Violates safety")
                x3 = xp_3 - l - min_s ### change safety constraints: guarantee no crash even when
                a_ = (x3 - x_ - v_*self.dt)*2/(self.dt**2)
                if a_<self.a_min or a_>self.a_max:
                    if(self.verbose):
                        print("CACC safety wrong acc")
                        #quit()
                v_ = v + s[(n)*3+2]*self.dt
                v3 = v_ + a_*self.dt
                if v3<self.v_min or v3>self.v_max:
                    if(self.verbose):
                        print("CACC safety wrong vel")
                        #quit()



        ######## update end here
        return v_, x_, a_

    def reward_function(self, states, disps):
        size = len(states)
        sum_squared_acc = 0
        #print("$$$")
        accels = []
        for s in range(size):
            state = []
            for i in range(self.num_vehicles):
                tmp = states[s][:3]
                states[s] = states[s][3:]
                state.append(tmp)

            # Calculate SUM of SQUARED ACCELERATION
            for n in range(self.num_vehicles - self.num_leading_cars):
                sum_squared_acc = sum_squared_acc + (1**n)*(state[self.num_leading_cars + n][2] ** 2)
                #print(n)
                accels.append((n,state[self.num_leading_cars + n][2]))
                #if(state[self.num_leading_cars + n][2] > 10 or state[self.num_leading_cars + n][2] < -10):
                    #print(accels[-1])
                    #print(self.t_start)
                    #print(len(accels)/(self.num_vehicles - self.num_leading_cars))
                    #input()

        # Calculate SUM of CAR DISPLACEMENT
        sum_car_disp = sum([(1**i)*disps[self.num_leading_cars:][i] for i in range(self.num_vehicles - self.num_leading_cars)])

        reward = sum_car_disp - self.LAMBDA * ((sum_squared_acc)/((self.num_vehicles - self.num_leading_cars)*size))
        #if((reward + sum(self.neg_rewards))/self.rew_normalize < -100):
            #print("$$$$")
            #print(sum_car_disp)
            #print(sum_squared_acc)
            #print(self.LAMBDA * ((sum_squared_acc)/((self.num_vehicles - self.num_leading_cars)*size)))
            #print(self.LAMBDA * ((self.num_vehicles - self.num_leading_cars)*size)/(sum_squared_acc + self.EPSILON) )
            #print(self.neg_rewards)
            #print(sum(self.neg_rewards))
            #print(reward)
            #print((reward + sum(self.neg_rewards)))
            #print((reward + sum(self.neg_rewards))/self.rew_normalize)
            #print("$$$$")
            #print(accels)
            #input()
            # quit()

        #return (reward)/self.rew_normalize
        #TODO #play with reward penalties in update state #Try DDPG
        self.results["reward"] = (reward + sum(self.neg_rewards))/self.rew_normalize
        self.results["reward_disp"] = sum_car_disp
        self.results["reward_fuel"] = -1 * self.LAMBDA * ((sum_squared_acc)/((self.num_vehicles - self.num_leading_cars)*size))
        self.results["reward_penalty"] = sum(self.neg_rewards)

        return (reward + sum(self.neg_rewards))/self.rew_normalize

    def reset(self):
        # Resets cumilators for this time frame
        #print("Environment Reset")
        self.t_start = randint(20,len(data)-self.max_t-1)
        break_index = [74,290,661,899,1619,1876,2668,2806,4323,4510,6655,6787,7044,7832,8128,8938,9962]
        while (True in [True for idx in break_index if (idx in range(self.t_start,self.t_start+self.max_t))==True]):
            self.t_start = randint(20,len(data)-self.max_t-1)
        self.current_states = [self.get_state(self.t_start)]
        self.current_disps = [0 for i in range(self.num_vehicles)]
        self.t = 0
        self.neg_rewards = []
        self.is_crashed = False

        self.results = {"min_gap": 0, "max_gap": 0, "violates_a": 0, "violates_v": 0, "reward": 0, "reward_disp": 0, "reward_fuel": 0, "reward_penalty": 0}
        return self.center_state(self.current_states[-1])

    def reset_from_state(self,t_start):
        # Resets cumilators for this time frame
        #print("Environment Reset")
        self.t_start = t_start
        #while (self.t_start > 4000 and self.t_start < 5000) or (self.t_start > 6000 and self.t_start < 7000) or (self.t_start > 8000 and self.t_start < 9000):
            #self.t_start = randint(20,len(data)-self.max_t-1)
        self.current_states = [self.get_state(self.t_start)]
        self.current_disps = [0 for i in range(self.num_vehicles)]
        self.t = 0
        self.neg_rewards = []
        self.is_crashed = False
        return self.center_state(self.current_states[-1])

    def render(self):
        # Prints Visulizations for us based on state
        self.win = pygame.display.set_mode((self.win_width,self.win_height))
        pygame.time.delay(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        width = self.win_width * 0.02 / self.num_vehicles
        height = self.win_height * 0.05 / self.num_vehicles
        start = self.current_states[0][3*(self.num_vehicles-1)+1]
        distance = self.max_t * 1
        scale = 0.25

        vel = 5

        self.win.fill((255,255,255))
        for n in range(self.num_leading_cars):
            pygame.draw.rect(self.win, (0, 0, 255), (((self.current_states[-1][3*n+1] - start)/distance) * self.win_width * scale + 50, self.win_height/2, width, height))
        pygame.draw.rect(self.win, (0, 255, 0), (((self.current_states[-1][3*(self.num_leading_cars)+1] - start)/distance) * self.win_width * scale + 50, self.win_height/2, width, height))
        for n in range(self.num_following_cars):
            pygame.draw.rect(self.win, (255, 0, 0), (((self.current_states[-1][3*(self.num_leading_cars+n+1)+1] - start)/distance) * self.win_width * scale + 50, self.win_height/2, width, height))

        pygame.display.update()
        os.system("clear")
        print("State")
        print()
        for n in range(self.num_vehicles):
            print("Car",n)
            color = None
            if (n < self.num_leading_cars):
                color = "Blue"
            elif (n == self.num_leading_cars):
                color = "Green"
            else:
                color = "Red"
            print("Color",color)
            print("Disp:",self.center_state(self.current_states[-1])[3*n+1])
            print("Vel:",self.current_states[-1][3*n+0])
            print("Acc:",self.current_states[-1][3*n+2])
            print()
        #print(self.center_state(self.current_states[-1]))

    def step(self, a, human=False, controller="OURS"):
        # returns next_state, reward, is_done, and info base on chosen acceleration
        s, a, r, s_, a_ = self.update_state(self.current_states[-1], a, human, controller)

        self.current_states.append(s_)
        self.neg_rewards.append(r)

        # Calc displacement
        for n in range(self.num_vehicles):
            self.current_disps[n] = self.current_disps[n] + (s_[n*3+1] - s[n*3+1])

        self.t = self.t + 1
        if self.t < self.max_t:
            if self.is_crashed:
                reward = self.reward_function(self.current_states, self.current_disps)
                return self.center_state(s_), reward, 1, {"CAV":0}
            else:
                return self.center_state(s_), 0, 0, {"CAV":0}
        else:
            reward = self.reward_function(self.current_states, self.current_disps)
            return self.center_state(s_), reward, 1, {"CAV":reward}
