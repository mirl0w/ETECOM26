import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from simulator import (SliceEnv, ACTIONS, N_PRB, SLICES, static_alloc,
                        proportional_fair_alloc, QLearningAgent)

SEED = 123
EPOCHS_PER_RUN = 500
N_RUNS = 20
LOADS = ['low', 'medium', 'high']

Q = np.load('q_table.npy')


def run_episode(method, load, rng):
    env = SliceEnv(load=load, rng=rng)
    s = env.reset()
    tot_arr = {sl: 0 for sl in SLICES}
    tot_served = {sl: 0 for sl in SLICES}
    tot_viol = {sl: 0 for sl in SLICES}
    tot_drop = {sl: 0 for sl in SLICES}
    tot_unused = 0
    waits = {sl: [] for sl in SLICES}

    for _ in range(EPOCHS_PER_RUN):
        if method == 'static':
            alloc = static_alloc(s)
        elif method == 'proportional_fair':
            alloc = proportional_fair_alloc(env)
        elif method == 'qlearning':
            a_idx = int(np.argmax(Q[s]))
            alloc = ACTIONS[a_idx]
        else:
            raise ValueError(method)

        s, r, info = env.step(alloc)
        for sl in SLICES:
            tot_arr[sl] += info['arrivals'][sl]
            tot_served[sl] += info['served'][sl]
            tot_viol[sl] += info['violations'][sl]
            tot_drop[sl] += info['drops'][sl]
            waits[sl].extend(info['wait'][sl])
        tot_unused += info['unused']

    urllc_violation_rate = tot_viol['URLLC'] / max(1, tot_arr['URLLC'])
    mmtc_drop_rate = tot_drop['mMTC'] / max(1, tot_arr['mMTC'])
    throughput_embb = tot_served['eMBB'] / EPOCHS_PER_RUN
    utilization = 1 - tot_unused / (N_PRB * EPOCHS_PER_RUN)
    avg_latency = {sl: (np.mean(waits[sl]) if waits[sl] else 0.0) for sl in SLICES}

    return dict(urllc_violation_rate=urllc_violation_rate, mmtc_drop_rate=mmtc_drop_rate,
                throughput_embb=throughput_embb, utilization=utilization,
                avg_latency=avg_latency, waits=waits)


def aggregate(method, load, n_runs=N_RUNS):
    rng = np.random.RandomState(SEED)
    keys = ['urllc_violation_rate', 'mmtc_drop_rate', 'throughput_embb', 'utilization']
    acc = {k: [] for k in keys}
    all_waits = {sl: [] for sl in SLICES}
    for i in range(n_runs):
        run_rng = np.random.RandomState(SEED + i)
        res = run_episode(method, load, run_rng)
        for k in keys:
            acc[k].append(res[k])
        for sl in SLICES:
            all_waits[sl].extend(res['waits'][sl])
    summary = {k: (float(np.mean(acc[k])), float(np.std(acc[k]))) for k in keys}
    return summary, all_waits


METHODS = ['static', 'proportional_fair', 'qlearning']
METHOD_LABEL = {'static': 'Static', 'proportional_fair': 'Proportional-Fair', 'qlearning': 'Q-learning'}

results = {}
waits_high_load = {}
for method in METHODS:
    results[method] = {}
    for load in LOADS:
        summary, waits = aggregate(method, load)
        results[method][load] = summary
        if load == 'high':
            waits_high_load[method] = waits['URLLC']

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ---------- Fig. 1: SLA (URLLC) violation rate vs. load ----------
plt.figure(figsize=(5, 4))
for method in METHODS:
    means = [results[method][load]['urllc_violation_rate'][0] * 100 for load in LOADS]
    stds = [results[method][load]['urllc_violation_rate'][1] * 100 for load in LOADS]
    plt.errorbar(LOADS, means, yerr=stds, marker='o', label=METHOD_LABEL[method], capsize=4)
plt.xlabel('Traffic load')
plt.ylabel('URLLC SLA violation rate (%)')
plt.title('SLA Violation Rate vs. Traffic Load')
plt.legend()
plt.tight_layout()
plt.savefig('fig1_sla_violation.png', dpi=150)
plt.close()

# ---------- Fig. 2: URLLC latency CDF at high load ----------
plt.figure(figsize=(5, 4))
for method in METHODS:
    w = np.sort(waits_high_load[method])
    if len(w) == 0:
        continue
    cdf = np.arange(1, len(w) + 1) / len(w)
    plt.plot(w, cdf, label=METHOD_LABEL[method])
plt.xlabel('URLLC packet wait time (epochs)')
plt.ylabel('CDF')
plt.title('URLLC Latency CDF under High Load')
plt.legend()
plt.tight_layout()
plt.savefig('fig2_latency_cdf.png', dpi=150)
plt.close()

# ---------- Fig. 3: training reward curve ----------
with open('reward_history.json') as f:
    reward_history = json.load(f)
reward_history = np.array(reward_history)
window = 50
smoothed = np.convolve(reward_history, np.ones(window) / window, mode='valid')
plt.figure(figsize=(5, 4))
plt.plot(reward_history, alpha=0.25, color='tab:blue', label='Raw episode reward')
plt.plot(np.arange(window - 1, len(reward_history)), smoothed, color='tab:blue', label=f'{window}-episode moving avg')
plt.xlabel('Training episode')
plt.ylabel('Total episode reward')
plt.title('Q-learning Training Convergence')
plt.legend()
plt.tight_layout()
plt.savefig('fig3_training_reward.png', dpi=150)
plt.close()

# ---------- Table I: averaged across all three loads ----------
table_rows = []
for method in METHODS:
    v = np.mean([results[method][load]['urllc_violation_rate'][0] for load in LOADS]) * 100
    d = np.mean([results[method][load]['mmtc_drop_rate'][0] for load in LOADS]) * 100
    t = np.mean([results[method][load]['throughput_embb'][0] for load in LOADS])
    u = np.mean([results[method][load]['utilization'][0] for load in LOADS]) * 100
    table_rows.append((METHOD_LABEL[method], v, d, t, u))

with open('table1.json', 'w') as f:
    json.dump(table_rows, f, indent=2)

print('Method            URLLC-viol%  mMTC-drop%  eMBB-thpt/epoch  PRB-util%')
for row in table_rows:
    print(f'{row[0]:<17} {row[1]:>10.2f}  {row[2]:>9.2f}  {row[3]:>14.2f}  {row[4]:>9.2f}')

# relative improvement of Q-learning vs static (headline numbers)
static_v = table_rows[0][1]
ql_v = table_rows[2][1]
static_u = table_rows[0][4]
ql_u = table_rows[2][4]
print('\nHeadline: URLLC violation reduction vs static: {:.1f}%'.format(
    100 * (static_v - ql_v) / static_v if static_v > 0 else float('nan')))
print('Headline: utilization change vs static: {:+.1f} percentage points'.format(ql_u - static_u))
