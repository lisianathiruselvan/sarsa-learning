# SARSA Learning Algorithm


## AIM

To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT

The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States:

The environment has 7 states:

    Two Terminal States: G: The goal state & H: A hole state.
    Five Transition states / Non-terminal States including S: The starting state.

### Actions:

The agent can take two actions:

    R: Move right.
    L: Move left.

### Transition Probabilities:

The transition probabilities for each action are as follows:

    50% chance that the agent moves in the intended direction.
    33.33% chance that the agent stays in its current state.
    16.66% chance that the agent moves in the opposite direction. For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards:
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

## SARSA LEARNING ALGORITHM


    Initialize the Q-values arbitrarily for all state-action pairs.

    Repeat for each episode:

i. Initialize the starting state.

ii. Repeat for each step of episode:

a. Choose action from state using policy derived from Q (e.g., epsilon-greedy).

b. Take action, observe reward and next state.

c. Choose action from next state using policy derived from Q (e.g., epsilon-greedy).

d. Update Q(s, a) := Q(s, a) + alpha * [R + gamma * Q(s', a') - Q(s, a)]

e. Update the state and action.

iii. Until state is terminal.

    Until performance converges.


## SARSA LEARNING FUNCTION
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    # Write your code here
    select_action = lambda state, Q, epsilon:np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])
      while not done:
        next_state, reward, done, _= env.step(action)
        next_action = select_action(next_state, Q, epsilons[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:

### Optimal Policy:

![326728154-43318948-a2a1-48f6-877b-7475305234a9](https://github.com/lisianathiruselvan/sarsa-learning/assets/119389971/7e1f188b-845e-4c8c-9b27-4c56d92fa2b3)

### First Visit Monte Carlo Method:

![326728232-9c2a0059-495f-411f-b85b-eb5a10675b1d](https://github.com/lisianathiruselvan/sarsa-learning/assets/119389971/b56fca58-9940-4de3-83d5-54e689425a07)

### SARSA Learning Algorithm:

![326728327-39d76c94-21c6-4f09-87d1-69d970873888](https://github.com/lisianathiruselvan/sarsa-learning/assets/119389971/9e9f42c5-70ca-4992-9d55-1abdae19299d)

### Plot for State Value Function - Monte Carlo:

![328122119-1e544588-a525-43f1-9e3a-d06f3ec603a3](https://github.com/lisianathiruselvan/sarsa-learning/assets/119389971/8b1d0d67-d486-4900-9fa0-88bfb7a36188)

### Plot for State Value Function - SARSA Learning:

![328122355-9b3d7dbe-cf87-4228-a953-840e41f368f5](https://github.com/lisianathiruselvan/sarsa-learning/assets/119389971/e585ef56-73e3-46d3-a9fc-aa930a0ba06a)

## RESULT:

Thus, the implementation of SARSA learning algorithm was implemented successfully.
