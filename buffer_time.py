"""
Numerical script for evaluating entanglement distribution rate in a simplest repeater with buffer time with 3 nodes (2 end nodes + 1 repeater node).

Each end node has N quantum memories (thus repeater node has 2N memories), when N > 1 entanglement purification will be included.

Memory decoherence is included. Comparison between dephasing and depolarizing channels will be made without operation imperfection.

Operation imperfection will be further included, and only memory depolarizing channel will be considered for this case.

Buffer time will be varied for optimization.
Entanglement distribution rate (average) will be calculated.
Probability distribution of distributed entangled state will be calculated.

@author: Allen Zang
"""

import numpy as np
from numpy.random import default_rng
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import time


"""analytical functions"""
def swap_fid_dp(f1, f2):
    """fidelity of dephased Bell state after swapping, without operation imperfection."""
    fid = f1*f2 + (1-f1)*(1-f2)
    
    return fid

def swap_fid_w(f1, f2):
    """fidelity of Werner state after swapping, without operation imperfection."""
    fid = f1*f2 + (1-f1)*(1-f2) / 3
    
    return fid

def swap_fid_w_ip(f1, f2, eta, p):
    """fidelity of Werner state after swapping, with operation imperfection."""
    e1 = (1-f1) / 3
    e2 = (1-f2) / 3
    fid = p * (eta**2 * (f1*f2 + 3*e1*e2) + (1 - eta**2)*(f1*e2 + e1*f2 + 2*e1*e2)) + (1-p) / 4
    
    return fid

def puri_fid_dp(f1, f2):
    """fidelity of dephased Bell state after purification, without operation imperfection."""
    fid = f1*f2 / (f1*f2 + (1-f1)*(1-f2))
    
    return fid

def puri_p_dp(f1, f2):
    """success probability of dephased Bell state purification, without operation imperfection."""
    prob = swap_fid_dp(f1, f2)
    
    return prob

def puri_fid_w(f1, f2):
    """fidelity of Werner state after purification, without operation imperfection."""
    fid = (f1*f2 + (1-f1)*(1-f2) / 9) / (f1*f2 + (f1*(1-f2) + (1-f1)*f2)/3 + 5*(1-f1)*(1-f2) / 9)
    
    return fid

def puri_p_w(f1, f2):
    """success probability of Werner state purification, without operation imperfection."""
    prob = f1*f2 + (f1*(1-f2) + (1-f1)*f2)/3 + 5*(1-f1)*(1-f2) / 9
    
    return prob

def puri_fid_w_ip(f1, f2, eta, p):
    """fidelity of Werner state after purification, with operation imperfection."""
    e1 = (1-f1) / 3
    e2 = (1-f2) / 3
    nu = (eta**2 + (1-eta)**2) * (f1*f2 + e1*e2) + 2 * eta * (1-eta) * (f1*e2 + e1*e2) + (1-p**2) / (8*p**2)
    de = (eta**2 + (1-eta)**2) * (f1*f2 + f1*e2 + e1*f2 + 5*e1*e2) + 2 * eta * (1-eta) * (2*f1*e2 + 2*e1*f2 + 4*e1*e2) + (1-p**2) / (2*p**2)
    fid = nu / de
    
    return fid

def puri_p_w_ip(f1, f2, eta, p):
    """success probability of Werner state purification, with operation imperfection."""
    e1 = (1-f1) / 3
    e2 = (1-f2) / 3
    prob = p**2 * (eta**2 + (1-eta)**2) * (f1*f2 + f1*e2 + e1*f2 + 5*e1*e2) + p**2 * 2 * eta * (1-eta) * (2*f1*e2 + 2*e1*f2 + 4*e1*e2) + (1-p**2) / 2
    
    return prob

def rains_bound(f):
    """Rains bound of distillable entanglement for Bell diagonal states."""
    de = 1 + f * np.log2(f) + (1-f) * np.log2(1-f)

    return de


"""entanglement link class"""
class Link():
    """Class of entanglement link.
    
    Attributes:
        fid_raw (float): raw fidelity upon generation
        beta (float): memory quality factor
        p_g (float): entanglement generation success probability, including photon loss
        form (str): denote if the state is dephased / depolarized Bell state
        rng: random generator to determine if entanglement generation is successful
    """
    def __init__(self, beta=1, p_g=1, fid_raw=1, seed=0, form="dephased"):
        assert 0 <= fid_raw <= 1, "Raw fidelity must be between 0 and 1."
        self.fid_raw = fid_raw
        
        assert 0 <= beta <= 1, "Memory quality factor must be between 0 and 1."
        self.beta = beta
        
        assert form == "dephased" or form == "werner", "State form must be either 'dephased' or 'werner'."
        self.form = form
        
        assert 0 <= p_g <= 1, "Entanglement generation hardware success probability must be between 0 and 1."  # arg p_s does not include photon loss
        self.p_g = p_g * 0.37  # consider 20km optical fiber for entanglement generation
        # self.p_g = p_g
        
        self.rng = default_rng(seed)

        self.fid = -1  # before successful EG, link fidelity is recorded as -1
        self.t_gen = -1  # before successful EG, time when link is generated is recorded as -1, will be used to determine purification pairing
        self.t_now = -1  # before successful EG, current time is recorded as -1, will be used to determine fidelity decay
        
    def ent_gen(self, t):
        """method for entanglement generation."""
        assert self.fid == -1 and self.t_gen == -1 and self.t_now == -1, "Entangled link has been established, should not invoke entanglement generation method."
        if self.rng.random() <= self.p_g:
            self.fid = self.fid_raw
            self.t_gen = t  # t is in units of tau (elementary link 1-cc time)
            self.t_now = t 
            
    def fid_decay(self, t):
        """method for memory decoherence-induced fidelity decay."""
        # t is simulation time in units of tau (elementary link 1-cc time)
        t_dec = t - self.t_now  # duration between now and last time when link fidelity was updated
        if self.form == "dephased":
            fid_new = self.fid * (1 + 2*self.beta**t_dec) / 3 + (1 - self.beta**t_dec) / 6
        elif self.form == "werner":
            fid_new = self.fid * self.beta**t_dec + (1 - self.beta**t_dec) / 4
            
        # update the (fidelity, time) pair of the link
        self.fid = fid_new
        self.t_now = t
        
    def purify_with(self, link, op_imperfect=False, p=-1, eta=-1):
        """method to purify the current link with another link, 
        i.e. the current link will not be measured, its fidelity corresponding to f1 in analytical formulae.
        if imperfection is included, {op_imperfect, p, eta} must be changed from the default values.
        """
        assert self.form == link.form, "Current simulation only supports purification of Bell states in same form, either 'dephased' or 'werner'."
        f1 = self.fid
        f2 = link.fid
        if self.form == "dephased":
            assert op_imperfect == False, "Current simulation does not support operation imperfection for dephased Bell state."
            fid_new = puri_fid_dp(f1, f2)
            p_succ = puri_p_dp(f1, f2)
            if self.rng.random() <= p_succ:
                self.fid = fid_new
            else:
                self.reset()
            link.reset()  # the other link which is measured will always be reset
        elif self.form == "werner":
            if op_imperfect == False:
                # not include operation imperfection
                fid_new = puri_fid_w(f1, f2)
                p_succ = puri_p_w(f1, f2)
                if self.rng.random() <= p_succ:
                    self.fid = fid_new
                else:
                    self.reset()
            else:
                # include operation imperfection
                assert p != -1 and eta != -1, "{p, eta} must be changed from the default values when operation imperfection is included."
                fid_new = puri_fid_w_ip(f1, f2, eta, p)
                p_succ = puri_p_w_ip(f1, f2, eta, p)
                if self.rng.random() <= p_succ:
                    self.fid = fid_new
                else:
                    self.reset()
            link.reset()  # the other link which is measured will always be reset

    def swap_with(self, link, p_s, op_imperfect=False, p=-1, eta=-1):
        """method to perform swapping on current link and another link, 
        if imperfection is included, {op_imperfect, p, eta} must be changed from the default values.
        """
        assert self.form == link.form, "Current simulation only supports swapping of Bell states in same form, either 'dephased' or 'werner'."
        f1 = self.fid
        f2 = link.fid
        if self.rng.random() > p_s:
            # after failed swapping two links are reset
            self.reset()
            link.reset()
            
            return -1
        else: 
            if self.form == "dephased":
                assert op_imperfect == False, "Current simulation does not support operation imperfection for dephased Bell state."
                fid_new = swap_fid_dp(f1, f2)
            elif self.form == "werner":
                if op_imperfect == False:
                    # not include operation imperfection
                    fid_new = swap_fid_w(f1, f2)
                else:
                    # include operation imperfection
                    assert p != -1 and eta != -1, "{p, eta} must be changed from the default values when operation imperfection is included."
                    fid_new = swap_fid_w_ip(f1, f2, eta, p)
            # after successful swapping distributed entangled state will not occupy memories for entnaglement generation
            self.reset()
            link.reset()
            
            return fid_new
        
    def reset(self):
        """method to reset entanglement link upon failed purification / swapping or destructive measurement."""
        self.fid = -1
        self.t_gen = -1
        self.t_now = -1


# global simulation parameters
NUM_TRIALS = 100000
MEMO_SIZE = 6  # maximal number of available quantum memories on one elementary link
MEMO_Q_FACTOR = 0.998  # memory quality factor, no greater than 1
GEN_HARDWARE_PROB = 0.1  # hardware success probability for entanglement generation, no greter than 1
RAW_FID = 1  # raw fidelity upon successful entanglement generation
SWAP_PROB = 0.5  # entanglement swapping success probability, no greter than 1
GATE_PROB = 0.9  # 2-qubit gate successs probability, no greter than 1
MEAS_PROB = 0.9  # 1-qubit measurement successs probability, no greter than 1
STATE_FORM = "werner"
BUFFER_TIME = 30  # maximal buffer time
OP_IMPERFECT = False  # if include operation imperfection
ENT_GEN_TIME = 1e-3  # time required for one attempt of entanglement generation, in s


"""Simulation time scheme
|---------------------------------------------------------- time step ----------------------------------------------------------|
|<-at the beginning of each time step,
|<-check if their exists multiple links already,
|<-if yes, do purification until at most one link remains
                                                          |<-after checking purification, generate entanglement
"""


"""main simulation function"""
def run_sim(t_buffer, p_s, left_links, right_links, op_imperfect=False, p_gate=-1, eta_meas=-1):
    """main simulation function
    
    Args:
        t_buffer (int): buffer time in units of elementary link 1-way cc time
        p_s (float): swapping success probability
        left_links (List[Link]): list of entanglement links objects for the left link of 1st level repeater
        right_links (List[Link]): list of entanglement links objects for the right link of 1st level repeater
        op_imperfect (Bool): if operation imperfection is considered; default False
        p_gate (float): two-qubit gate no-error probability; default -1
        eta_meas (float): single-qubit measurement no-error probability; default -1
        
    Return:
        fid_dist (float): fidelity of distributed entangled pair; if swapping is unsuccessful the return value is -1
    """
    assert len(left_links) == len(right_links), "Both sides should have equal number of available memories."
    # simulation time initialization
    t = 0
    # established links list initialization, for organization of purification and swapping
    left_ready_links = []
    right_ready_links = []
    
    while t < t_buffer + 1:
        # perform purification if multiple links are already available (by the end of last time unit)
        for link in left_ready_links:
            assert link.t_gen != -1, "Ready left link must be really generated."
        for link in right_ready_links:
            assert link.t_gen != -1, "Ready right link must be really generated."
        
        if len(left_ready_links) > 1:
            # first invoke fidelity decay if multiple (>1) links are ready because the fidelities of states as input to purification has decayed since their generation
            for link in left_ready_links:
                link.fid_decay(t)
            # there can be 2 or more ready links
            # if there are more than 2, the only possibility is that 2 or more links are generated during last time step
            left_links_old = []  # link not generated during last time step
            for link in left_ready_links:
                if link.t_gen < t-1:
                    left_links_old.append(link)
            assert len(left_links_old) <= 1, "There must be at most one left ready link generated earlier than last time step."

            left_links_new = []
            for link in left_ready_links:
                if link.t_gen == t-1:
                    left_links_new.append(link)
            assert len(left_links_new) + len(left_links_old) == len(left_ready_links), "Total number of left ready links incompatible."

            # then first choose one of the last generated links to get purified with the old link, and secondly successively use remaining new links to purify until atmost one ready link is left
            # note the possibility of purification failure, after each purification attempt the ready links list need update
            while len(left_ready_links) > 1:
                left_link_new = left_links_new[0]
                if len(left_links_old) > 0:
                    left_link_old = left_links_old[0]
                    left_link_new.purify_with(left_link_old, op_imperfect=op_imperfect, p=p_gate, eta=eta_meas)
                    left_ready_links.remove(left_link_old)  # remove old link
                    left_links_old = []  # clear after purifying new link
                    if left_link_new.t_gen == -1:
                        # if purification failed, remove the new link as well
                        left_links_new.remove(left_link_new)
                        left_ready_links.remove(left_link_new)
                else:
                    left_link_new_2 = left_links_new[1]
                    left_link_new.purify_with(left_link_new_2, op_imperfect=op_imperfect, p=p_gate, eta=eta_meas)
                    # remove the second new link
                    left_ready_links.remove(left_link_new_2)
                    left_links_new.remove(left_link_new_2)
                    if left_link_new.t_gen == -1:
                        # if purification failed, remove the new link as well
                        left_links_new.remove(left_link_new)
                        left_ready_links.remove(left_link_new)

        # do the above again for right links
        if len(right_ready_links) > 1:
            for link in right_ready_links:
                link.fid_decay(t)
        
            right_links_old = []  # link not generated during last time step
            for link in right_ready_links:
                if link.t_gen < t-1:
                    right_links_old.append(link)
            assert len(right_links_old) <= 1, "There must be at most one right ready link generated earlier than last time step."

            right_links_new = []
            for link in right_ready_links:
                if link.t_gen == t-1:
                    right_links_new.append(link)
            assert len(right_links_new) + len(right_links_old) == len(right_ready_links), "Total number of right ready links incompatible."

            while len(right_ready_links) > 1:
                right_link_new = right_links_new[0]
                if len(right_links_old) > 0:
                    right_link_old = right_links_old[0]
                    right_link_new.purify_with(right_link_old, op_imperfect=op_imperfect, p=p_gate, eta=eta_meas)
                    right_ready_links.remove(right_link_old)  # remove old link
                    right_links_old = []  # clear after purifying new link
                    if right_link_new.t_gen == -1:
                        # if purification failed, remove the new link as well
                        right_links_new.remove(right_link_new)
                        right_ready_links.remove(right_link_new)
                else:
                    right_link_new_2 = right_links_new[1]
                    right_link_new.purify_with(right_link_new_2, op_imperfect=op_imperfect, p=p_gate, eta=eta_meas)
                    # remove the second new link
                    right_ready_links.remove(right_link_new_2)
                    right_links_new.remove(right_link_new_2)
                    if right_link_new.t_gen == -1:
                        # if purification failed, remove the new link as well
                        right_links_new.remove(right_link_new)
                        right_ready_links.remove(right_link_new)
        
        # invoke ent_gen if link not established
        if t < t_buffer:  #last time step (t = t_buffer) is only for finishing final purifications if still multiple links available, will not generate entanglement
            for link in left_links:
                if link.fid == -1 and link.t_gen == -1:
                    link.ent_gen(t)
                    if link.t_gen != -1:
                        left_ready_links.append(link)  # if successfully generated add to ready link list
            
            for link in right_links:
                if link.fid == -1 and link.t_gen == -1:
                    link.ent_gen(t)
                    if link.t_gen != -1:
                        right_ready_links.append(link)  # if successfully generated add to ready link list
        
        # update simulation time
        t += 1

    assert len(left_ready_links) <= 1 and len(right_ready_links) <= 1, "After buffer time, there should be at most one link available on both sides."
    if len(left_ready_links) ==  1 and len(right_ready_links) == 1:
        left_link_final = left_ready_links[0]
        right_link_final = right_ready_links[0]
        # perform final fidelity decay before swapping
        left_link_final.fid_decay(t_buffer+1)
        right_link_final.fid_decay(t_buffer+1)
        fid_dist = left_link_final.swap_with(right_link_final, p_s, op_imperfect=op_imperfect, p=p_gate, eta=eta_meas)
    else:
        fid_dist = -1  # swapping cannot be performed due to lack of ready link

    # in the end return the value of distributed fidelity for one single cycle of repeater, will be used to study performance statistics
    return fid_dist


"""test run simulation"""
"""
fid_res = []  # list of distributed states' fidelity over all trials as result

tick = time.time()
for trial in range(NUM_TRIALS):
    # set up links 
    seed_start = MEMO_SIZE * 2 * trial  # seed for rng of links
    left_links = [Link(beta=MEMO_Q_FACTOR, p_g=GEN_HARDWARE_PROB, fid_raw=RAW_FID, seed=seed_start+i, form=STATE_FORM) for i in range(MEMO_SIZE)]
    right_links = [Link(beta=MEMO_Q_FACTOR, p_g=GEN_HARDWARE_PROB, fid_raw=RAW_FID, seed=seed_start+MEMO_SIZE+i, form=STATE_FORM) for i in range(MEMO_SIZE)]

    # call the main simulation function
    fid_dist = run_sim(BUFFER_TIME, SWAP_PROB, left_links, right_links, op_imperfect=OP_IMPERFECT, p_gate=GATE_PROB, eta_meas=MEAS_PROB)  # fidelity of distributed state's fidelity in this trial
    fid_res.append(fid_dist)

assert len(fid_res) == NUM_TRIALS, "The number of results should be equal to the number of simulation trials."

sim_time = time.time() - tick
print(f"Time taken for {NUM_TRIALS} trials of repeater with {BUFFER_TIME} buffer time and {MEMO_SIZE} available memories: {(sim_time)*10**3:.03f}ms")
"""


"""plotting routine"""
plt.rc('font', size=10)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=15)

plt.rcParams['axes.titley'] = 1.05
plt.rcParams['axes.titlepad'] = 0

fig = plt.figure(figsize=(8, 6))
fig2 = plt.figure(figsize=(8, 6))


"""result visualization"""
# plt.title("average rate v.s. buffer time")
# plt.xlabel("buffer time (in $\mathrm{ms}$)")
# plt.ylabel("average rate (in $\mathrm{s}^{-1}$)")

ax = fig.add_subplot(111)  # for visualization of both optimal rate and optimized buffer time in one figure
ax.set_xlabel("memory number")
ax.set_ylabel("entanglement rate (in $\mathrm{s}^{-1}$)")
ax2 = ax.twinx()
ax2.set_ylabel("time (in $\mathrm{ms}$)")

rate_opt_list = []  # list initialization for optimal rate vs different memory number
rate_opt_per_memo_list = []  # list initialization for optimal rate per memory vs different memory number
t_buffer_opt_list = []  # list initialization for optimal buffer time vs different memory number

memo_num_list = np.arange(1, MEMO_SIZE, 1, dtype=int)  # varying number of available quantum memories
t_buffer_list = np.arange(1, BUFFER_TIME, 1, dtype=int)  # varying buffer time

for memo_size in memo_num_list:
    rate_list = []  # initialization of list of rate vs varying buffer time for a certian memory number
    for t_buffer in t_buffer_list:
        fid_res = []  # list of distributed states' fidelity over all trials as result

        tick = time.time()
        for trial in range(NUM_TRIALS):
            # set up links 
            seed_start = memo_size * 2 * trial  # seed for rng of links
            left_links = [Link(beta=MEMO_Q_FACTOR, p_g=GEN_HARDWARE_PROB, fid_raw=RAW_FID, seed=seed_start+i, form=STATE_FORM) for i in range(memo_size)]
            right_links = [Link(beta=MEMO_Q_FACTOR, p_g=GEN_HARDWARE_PROB, fid_raw=RAW_FID, seed=seed_start+memo_size+i, form=STATE_FORM) for i in range(memo_size)]

            # call the main simulation function
            fid_dist = run_sim(t_buffer, SWAP_PROB, left_links, right_links, op_imperfect=OP_IMPERFECT, p_gate=GATE_PROB, eta_meas=MEAS_PROB)  # fidelity of distributed state's fidelity in this trial
            fid_res.append(fid_dist)

        assert len(fid_res) == NUM_TRIALS, "The number of results should be equal to the number of simulation trials."

        sim_time = time.time() - tick
        print(f"Time taken for {NUM_TRIALS} trials of repeater with {t_buffer} buffer time and {memo_size} available memories: {(sim_time)*10**3:.03f}ms")

        # calculate average rate for this specific choice of memory number and buffer time 
        p_succ = sum(fid >= 0 for fid in fid_res) / len(fid_res)  # success probability
        if p_succ == 0:
            rate = 0
        else:
            fid_avg = sum(fid for fid in fid_res if fid >= 0) / sum(fid >= 0 for fid in fid_res)  # average fidelity of distributed state
            if fid_avg < 0.5:
                rate = 0
            else: 
                dist_ent = rains_bound(fid_avg)  # distillable entanglement upper bound of average distributed state from Rains bound
                buffer_time = t_buffer * ENT_GEN_TIME  # buffer time in s
                rate = p_succ * dist_ent / buffer_time  # average rate

        rate_list.append(rate)

    rate_opt = max(rate_list)  # maximal entanglement rate
    rate_opt_per_memo = rate_opt / memo_size  # maximal entanglement rate per memory
    rate_opt_list.append(rate_opt)
    rate_opt_per_memo_list.append(rate_opt_per_memo)
    t_buffer_opt = np.argmax(np.array(rate_list)) + 1  # buffer time that maximizes rate, i.e. optimized buffer time  # note that np.argmax() only counts the first occurrence
    t_buffer_opt_list.append(t_buffer_opt)

#     plt.plot(t_buffer_list*ENT_GEN_TIME, rate_list, label='$M={}$'.format(memo_size))
# plt.legend()

print(t_buffer_opt_list)  # print out optimized buffer time for plotting of probability distribution of fidelity of distributed entanglement
ax.plot(memo_num_list, rate_opt_list, label='rate')
ax.plot(memo_num_list, rate_opt_per_memo_list, label='rate per memo')
ax2.plot(memo_num_list, t_buffer_opt_list, '--g', label='buffer time')
ax.legend(loc=0)
ax2.legend(loc=0)

# plt.show()


# plot fidelity probability distribution (density plot with Gaussian kernel)
ax3 = fig2.add_subplot(111)  # for visualization of fidelity probability distribution
ax3.set_xlabel("fidelity")
ax3.set_xlim(0.5, 1)
ax3.set_ylabel("density")
for idx, memo_size in enumerate(memo_num_list):
    opt_buffer_time = t_buffer_opt_list[idx]  # optimized buffer time corresponding to memory size
    fid_res = []  # list of distributed states' fidelity over all trials as result

    tick = time.time()
    for trial in range(NUM_TRIALS):
        # set up links 
        seed_start = memo_size * 2 * trial  # seed for rng of links
        left_links = [Link(beta=MEMO_Q_FACTOR, p_g=GEN_HARDWARE_PROB, fid_raw=RAW_FID, seed=seed_start+i, form=STATE_FORM) for i in range(memo_size)]
        right_links = [Link(beta=MEMO_Q_FACTOR, p_g=GEN_HARDWARE_PROB, fid_raw=RAW_FID, seed=seed_start+memo_size+i, form=STATE_FORM) for i in range(memo_size)]

        # call the main simulation function
        fid_dist = run_sim(opt_buffer_time, SWAP_PROB, left_links, right_links, op_imperfect=OP_IMPERFECT, p_gate=GATE_PROB, eta_meas=MEAS_PROB)  # fidelity of distributed state's fidelity in this trial
        fid_res.append(fid_dist)

    assert len(fid_res) == NUM_TRIALS, "The number of results should be equal to the number of simulation trials."

    sim_time = time.time() - tick
    print(f"Time taken for {NUM_TRIALS} trials of repeater with {opt_buffer_time} buffer time and {memo_size} available memories: {(sim_time)*10**3:.03f}ms")

    fid_succ = np.array([fid for fid in fid_res if fid >= 0])  # list of fidelities of successfully distributed entangled pair
    density = gaussian_kde(fid_succ)
    fid_list = np.linspace(0, 1, num=100)

    ax3.plot(fid_list, density(fid_list), label='$M={}$'.format(memo_size))

ax3.legend()

plt.show()






