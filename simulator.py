import numpy as np
import random, json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_PRB = 50
SLICES = ['eMBB', 'URLLC', 'mMTC']
DEADLINE = {'eMBB': 10, 'URLLC': 1, 'mMTC': 5}   # epochs before a queued packet expires
BASE_LAMBDA = {'eMBB': 15.0, 'URLLC': 6.0, 'mMTC': 10.0}  # mean arrivals/epoch at "medium" load
LOAD_SCALE = {'low': 0.6, 'medium': 1.1, 'high': 1.8}

# Discrete PRB allocation profiles available to the RL agent (sum to N_PRB)
ACTIONS = [
    (30, 10, 10), (25, 15, 10), (20, 20, 10), (15, 25, 10), (10, 30, 10),
    (20, 15, 15), (15, 20, 15), (10, 20, 20), (20, 10, 20), (35, 10, 5),
]
for a in ACTIONS:
    assert sum(a) == N_PRB, a

W1, W2, W3, W4 = 1.0, 5.0, 1.0, 0.5   # reward weights (throughput, URLLC violation, mMTC drop, waste)
STATE_BINS = 5   # queue occupancy discretized into 0..4 per slice
STATE_CAP = 20    # queue length capped at this value before binning


def discretize(q):
    return min(q, STATE_CAP) * (STATE_BINS - 1) // STATE_CAP


class SliceEnv:
    """Single-gNB, 3-slice resource allocation environment."""

    def __init__(self, load='medium', rng=None):
        self.load = load
        self.rng = rng if rng is not None else np.random.RandomState(SEED)
        self.reset()

    def reset(self):
        self.queues = {s: [] for s in SLICES}
        self.t = 0
        return self.state()

    def state(self):
        return tuple(discretize(len(self.queues[s])) for s in SLICES)

    def _arrivals(self):
        scale = LOAD_SCALE[self.load]
        out = {}
        for s in SLICES:
            lam = BASE_LAMBDA[s] * scale
            n = self.rng.poisson(lam)
            if s == 'URLLC' and self.rng.rand() < 0.15:   # occasional burst
                n += self.rng.poisson(lam * 3)
            out[s] = n
        return out

    def step(self, alloc):
        """alloc: tuple of PRBs per slice (eMBB, URLLC, mMTC), must sum to N_PRB."""
        arrivals = self._arrivals()
        for s in SLICES:
            self.queues[s].extend([self.t] * arrivals[s])

        served, violations, drops, wait_logged = {}, {}, {}, {}
        unused = 0
        for i, s in enumerate(SLICES):
            q = self.queues[s]
            deadline = DEADLINE[s]
            expired = [a for a in q if self.t - a > deadline]
            q = [a for a in q if self.t - a <= deadline]
            n_expired = len(expired)

            cap = alloc[i]
            n_serve = min(cap, len(q))
            served_packets = q[:n_serve]
            remaining = q[n_serve:]
            unused += max(0, cap - len(q))

            self.queues[s] = remaining
            served[s] = n_serve
            wait_logged[s] = [self.t - a for a in served_packets]

            if s == 'URLLC':
                violations[s], drops[s] = n_expired, 0
            elif s == 'mMTC':
                violations[s], drops[s] = 0, n_expired
            else:
                violations[s], drops[s] = 0, 0   # eMBB: expired packets are just aged out, not penalized

        self.t += 1
        thr_norm = served['eMBB'] / (BASE_LAMBDA['eMBB'] * LOAD_SCALE[self.load] + 1e-6)
        v_urllc = violations['URLLC']
        d_mmtc = drops['mMTC']
        u_unused = unused / N_PRB
        reward = W1 * thr_norm - W2 * v_urllc - W3 * d_mmtc - W4 * u_unused

        info = dict(arrivals=arrivals, served=served, violations=violations,
                    drops=drops, unused=unused, wait=wait_logged)
        return self.state(), reward, info


def static_alloc(_state):
    return (25, 15, 10)   # fixed 3GPP-style 50/30/20% split of N_PRB=50


def proportional_fair_alloc(env):
    qs = {s: len(env.queues[s]) for s in SLICES}
    total = sum(qs.values())
    if total == 0:
        base = N_PRB // 3
        alloc = [base, base, N_PRB - 2 * base]
    else:
        raw = {s: N_PRB * qs[s] / total for s in SLICES}
        alloc = [max(2, int(round(raw[s]))) for s in SLICES]
        diff = N_PRB - sum(alloc)
        alloc[0] += diff   # fix rounding drift on eMBB
    return tuple(alloc)


class QLearningAgent:
    def __init__(self, alpha=0.15, gamma=0.9, eps_start=1.0, eps_end=0.05, eps_decay_episodes=1500):
        self.Q = np.zeros((STATE_BINS, STATE_BINS, STATE_BINS, len(ACTIONS)))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_episodes = eps_start, eps_end, eps_decay_episodes

    def epsilon(self, episode):
        frac = min(1.0, episode / self.eps_decay_episodes)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state, episode=None, greedy=False):
        if not greedy and random.random() < self.epsilon(episode or 0):
            return random.randrange(len(ACTIONS))
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next):
        best_next = np.max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])


def train_q_learning(n_episodes=3000, epoch_per_episode=200):
    agent = QLearningAgent(eps_decay_episodes=int(n_episodes * 0.5))
    reward_history = []
    rng = np.random.RandomState(SEED)
    loads = ['low', 'medium', 'high']
    for ep in range(n_episodes):
        load = loads[ep % 3]  # cycle through load regimes for generalization
        env = SliceEnv(load=load, rng=rng)
        s = env.reset()
        ep_reward = 0.0
        for _ in range(epoch_per_episode):
            a = agent.act(s, episode=ep)
            s_next, r, _ = env.step(ACTIONS[a])
            agent.update(s, a, r, s_next)
            s = s_next
            ep_reward += r
        reward_history.append(ep_reward)
    return agent, reward_history


if __name__ == '__main__':
    agent, reward_history = train_q_learning()
    np.save('q_table.npy', agent.Q)
    with open('reward_history.json', 'w') as f:
        json.dump(reward_history, f)
    print('Training complete. Final 100-episode avg reward:', np.mean(reward_history[-100:]))
