import matplotlib
matplotlib.use('Agg')

import os.path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

import json

from Game import Game
from Rules import Rules
from Player import Player

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#=======================================================================================================================
# BENCHMARK PARAMETERS
shuffle_turns = True
# agent_eps = np.array([0.3])
# agent_half_life = np.array([1000, 2000, 3000, 5000, 8000])
N = 1000

# SAVE PARAMETERS
path = f'/home/benny/Code/Artificial_Intelligence/Dice10000/results/{now_str}'
if not os.path.exists(path):
    os.makedirs(path)
#=======================================================================================================================

benchmark_params = {
    # 'shuffle_turns': shuffle_turns,
    # 'agent_eps': agent_eps.tolist(),
    # 'agent_half_life': agent_half_life.tolist(),
    'N': N
}

with open(os.path.join(path, 'params.json'), 'w') as outf:
    json.dump(benchmark_params, outf, sort_keys=True, indent=4)

rules = Rules()
# N_eps = agent_eps.shape[0]
# N_hl = agent_half_life.shape[0]

# players = [Player(f'Player{i*N_hl+j} (eps={agent_eps[i]:.3f}, HL={agent_half_life[j]})',
#                   'score_weighted', rules, eps=agent_eps[i], half_life_score=agent_half_life[j])
#            for i in range(agent_eps.shape[0]) for j in range(agent_half_life.shape[0])]

players = [
    # Player('Player1', 'optimal_expected', rules=rules, log_path=os.path.join(path, 'log_actions.log')),
    Player('Player1', 'optimal_expected', rules=rules),
    Player('Player2', 'greedy', rules=rules, eps=0.),
    # Player('Player3', 'greedy', rules=rules, eps=0.3)
]

game = Game(players, rules)

game_runtimes = np.zeros([N])
wins = np.zeros([len(players), N])

t0 = time.time()
prev_time = t0

for i in range(N):
    if shuffle_turns:
        perm = np.random.permutation(len(players))
        print(f'\r{i+1}/{N}', end='')
        game = Game(list(np.array(players)[perm]), rules, render_mode=None)
        winner = game.play(step_by_step=False, interactive=False)
        wins[perm[winner], i] = 1

    else:
        print(f'\r{i + 1}/{N}', end='')
        game = Game(players, rules, render_mode=None)
        winner = game.play(step_by_step=False, interactive=False)
        wins[winner, i] += 1

    curr_time = time.time()
    game_runtimes[i] = curr_time - prev_time
    prev_time = curr_time

stat_str = f'Statistics after {N} games\n'
for i in range(len(players)):
    wins_i = np.sum(wins[i])
    stat_str += f'{players[i].name}: {wins_i} wins ({wins_i/N*100:.3f}%)\n'

with open(os.path.join(path, f'{now_str}_stats.txt'), 'w') as outf:
    outf.write(stat_str)

plt.figure()
plt.plot(game_runtimes*1000)
plt.ylabel('Elapsed time [ms]')
plt.xlabel('Number of simulated games')
plt.grid()
plt.savefig(os.path.join(path, f'{now_str}_compute_times.png'), bbox_inches='tight', dpi='figure')

plt.figure()
for i in range(len(players)):
    plt.plot(np.cumsum(wins[i]), label=f'{players[i].name}')
plt.xlabel('Number of simulated games')
plt.ylabel('Cumulated wins')
plt.grid()
plt.legend()
plt.savefig(os.path.join(path, f'{now_str}_cumulated_wins.png'), bbox_inches='tight', dpi='figure')

plt.figure()
plt.bar(np.arange(len(players)), np.sum(wins, axis=1)/N)
plt.xlabel('Player ID')
plt.ylabel('Probability of winning')
plt.grid()
plt.savefig(os.path.join(path, f'{now_str}_winning_probs.png'), bbox_inches='tight', dpi='figure')
# plt.show()
