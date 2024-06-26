
import numpy as np
import random
import math
import time
from tqdm import tqdm


'''
TimeGeo Model
'''

class Time_geo(object):

    def __init__(self, region_input, pop_input, p_t_raw=None, pop_num=7, time_slot=10, rho=0.6, gamma=0.41, alpha=1.86, n_w=6.1, beta1=3.67, beta2=10, simu_slot=144):
        
        super().__init__()
        self.time_slot = time_slot # time resolution is half an hour
        self.rho = rho # it controls the exploration probability for other regions
        self.gamma = gamma # it is the attenuation parameter for exploration probability
        self.alpha = alpha # it controls the exploration depth
        self.n_w = n_w # it is the average number of tour based on home a week.
        self.beta1 = beta1 # dwell rate
        self.beta2 = beta2 # burst rate
        self.simu_slot = simu_slot
        self.pop_num = pop_num

        self.sample_region = region_input
        # p_t_raw = p_t_raw if p_t_raw is not None else np.load('./timegeo/rhythm.npy', allow_pickle=True)
        p_t_raw = p_t_raw if p_t_raw is not None else np.load('Code/data/rhythm.npy', allow_pickle=True)
        
        
        self.p_t = np.array(p_t_raw).reshape(-1, (time_slot // 10)).sum(axis=1)
        self.region_num = self.sample_region.shape[0]
        self.home_location = np.random.choice(len(pop_input), pop_num, p=pop_input)
        self.pop_info = self.trace_simulate()

    def distance(self, p1, p2):#caculate distance
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def get_p_t(self, now_time):
        now_time_tup = time.localtime(float(now_time))
        i = int((now_time_tup.tm_wday * 24 * 60 + now_time_tup.tm_hour * 60 + now_time_tup.tm_min) / self.time_slot)
        return self.p_t[i]
    
    def predict_next_place_time(self, p_t_value, current_location_type):
        p1 = 1 - self.n_w * p_t_value
        p2 = 1 - self.beta1 * self.n_w * p_t_value
        p3 = self.beta2 * self.n_w * p_t_value
        location_is_change = 0
        new_location_type = 'undefined'
        if current_location_type == 'home':
            if random.uniform(0, 1) <= p1:
                new_location_type = 'home'
                location_is_change = 0
            else:
                new_location_type = 'other'
                location_is_change = 1
        elif current_location_type == 'other':
            p = random.uniform(0, 1)
            if p <= p2:
                new_location_type = 'other'
                location_is_change = 0
            elif random.uniform(0, 1) <= p3:
                new_location_type = 'other'
                location_is_change = 1
            else:
                new_location_type = 'home'
                location_is_change = 1
        if new_location_type == 'home':
            return 0, location_is_change
        else:
            return 2, location_is_change
        
    def negative_pow(self, k):
        p_k = {}
        for i, region in enumerate(k, 1):
            p_k[region[0]] = i ** (-self.alpha)
        temp = sum(p_k.values())
        for key in p_k:
            p_k[key] = p_k[key] / temp
        return p_k

    def predict_next_place_location_simplify(self, P_new, region_history, current_region, home_region):
        rp = random.uniform(0, 1)
        prob_accum, next_region = 0, 0
        if random.uniform(0, 1) < P_new:
            # explore; the explore distance is depend on history->delta_r
            length = {}
            for i, cen in enumerate(self.sample_region):
                if i in region_history:
                    continue
                length[i] = self.distance(cen, self.sample_region[current_region])
            try:
                del length[home_region]
                del length[current_region]
            except KeyError:
                pass
            k = sorted(length.items(), key=lambda x: x[1], reverse=False)
            p_k = self.negative_pow(k)
            for i, key in enumerate(p_k):
                prob_accum += p_k[key]
                if prob_accum > rp:
                    next_region = key
                    region_history[key] = 1
                    break
                else:
                    continue
        else:
            # return
            region_history_sum = sum(region_history.values())
            for key in region_history:
                prob_accum += region_history[key]/region_history_sum
                if rp < prob_accum:
                    next_region = key
                    region_history[key] += 1
                    break
        return next_region

    def predict_next_place_location(self, region_history, current_location, home_region):
        s = len(region_history.values())
        p_new = self.rho * s ** (-self.gamma) if s != 0 else 1
        return self.predict_next_place_location_simplify(p_new, region_history, current_location, home_region)

    def individual_trace_simulate(self, info, start_time, simu_slot):
        current_location_type = 'home'
        simu_trace = [[info['home'],start_time]]
        for i in range(simu_slot - 1):
            # pt is the currently move based probability
            now_time = (i+1) * 60 * self.time_slot + start_time
            p_t_value = self.get_p_t(now_time)
            now_type, location_change = self.predict_next_place_time(p_t_value, current_location_type)
            if location_change == 1:
                current_location = simu_trace[-1][0]
                if now_type == 0:
                    next_location = info['home']
                    current_location_type = 'home'
                else:
                    next_location = self.predict_next_place_location(info['region_history'], current_location, info['home'])
                    current_location_type = 'other'
                info['feature']['move_num'] += 1
                info['feature']['move_distance'] += self.distance(self.sample_region[next_location], self.sample_region[current_location])
            else:
                next_location = simu_trace[-1][0]
            simu_trace.append([next_location, now_time])
        return simu_trace
    
    def trace_simulate(self):
        pop_info = []
        for i in range(self.pop_num):
            pop_info.append({'n_w': self.n_w, 'beta1': self.beta1, 'beta2': self.beta2, 'home': self.home_location[i],
                            'feature': {'move_num': 0, 'move_distance': 0}, 'region_history': {}})
            pop_info[i]['trace'] = np.array(self.individual_trace_simulate(pop_info[i], 1621785600, self.simu_slot))
        return pop_info

def padding(traj, tim_size):
    def intcount(seq):
        a, b = np.array(seq[:-1]), np.array(seq[1:])
        return (a == a.astype(int)) + np.ceil(b) - np.floor(a) - 1
    locs = np.concatenate(([-1], traj['loc'], [-1]))
    tims = np.concatenate(([0], traj['tim'] % tim_size, [tim_size]))
    tims[-2] = tims[-1] if (tims[-2] < tims[-3]) else tims[-2]
    return np.concatenate([[locs[id]] * int(n) for id, n in enumerate(intcount(tims))]).astype(int)

def fixed(pad_traj, slot = 30):
    return np.array([np.argmax(np.bincount((pad_traj + 1)[(slot*i):(slot*i+slot)])) - 1 for i in range(int(len(pad_traj)/slot))])

def to_fixed(traj, tim_size, slot = 30):
    a = fixed(padding(traj, tim_size), slot)
    return np.where(a==-1, a[-1], a)

def to_std(traj, tim_size, detrans, time_slot=10):
    id = np.append(True, traj[1:] != traj[:-1])
    loc, tim = np.array(list(map(detrans,  traj[id]))), np.arange(0, tim_size, time_slot)[id]
    sta = np.append(tim[1:], tim_size) - tim
    return {'loc': loc, 'tim':tim, 'sta': sta}



def TimeGeo(data, param):
    TG = {}
    gen_bar = tqdm(data.items())
    for uid, trajs in gen_bar:
        gen_bar.set_description("TimeGeo - Generating trajectories for user: {}".format(uid))

        locations = np.sort(np.unique(np.concatenate([trajs[traj]['loc'] for traj in trajs])))
        trans = lambda x:np.where(locations == x)[0][0]
        detrans = lambda x:locations[x]

        input = np.array([to_fixed({'loc': list(map(trans, traj['loc'])), 'tim': traj['tim'], 'sta': traj['sta']}, param.tim_size, 10) for traj in trajs.values()])
        time_geo = Time_geo(param.GPS[np.ix_(locations)], np.bincount(input.flatten()) / np.cumprod(input.shape)[-1], simu_slot=param.tim_size//10)
        TG[uid] = {id: to_std(r['trace'][:, 0], param.tim_size, detrans) for id, r in enumerate(time_geo.pop_info)}
 
    return TG


