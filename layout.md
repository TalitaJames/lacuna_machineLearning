# Information for portfolio:
<!-- TODO: flowchart for portfolio -->

## Abstract
[Lacuna](https://www.cmyk.games/products/lacuna) is a two-player,
perfect information board game where flowers are randomly distributed in a continuous space.
Players alternate turns, strategically placing their tokens to claim the most flowers.

This project explores the development and evaluation of two deep reinforcement learning (RL) agents, PPO and SAC to compete at this game. Our objective was to investigate the effectiveness of modern RL algorithms in a spatial, adversarial game setting.



## Video
<iframe width="560" height="315"
src="https://www.youtube.com/watch?v=0Wa6__TQJ3M frameborder="0" allowfullscreen>
</iframe>

## Approach
We implemented two AI models, each extending from a common player base class.
As the state and action space is continious,
Deep RL algorithms were decidedly the most effective.

### Proximal Policy Optimization (PPO)
- PPO was implemented as a baseline RL agent.
- The agent struggled to learn effective strategies in the Lacuna environment, likely due to the continuous and adversarial nature of the action space.

### Soft Actor-Critic (SAC)
- The SAC agent demonstrated a better understanding of the game mechanics and consistently played valid moves.
- However, neither agent achieved a win rate significantly above 50%, suggesting that both agents learned at a similar pace.
- The results also indicate a potential inherent bias in the game favoring the second player, as observed in self-play experiments.

### PPO
- Struggled to

### SAC
- Played the game properly
- Didn't win over 50%
- Think this is because they both learnt more
- and the game is inherently biased towards the last player

## Results
<!-- What happened? -->


<!-- Future work? -->

<!-- 
## Results

- Both agents were able to learn the rules and valid actions of Lacuna through self-play.
- The SAC agent outperformed PPO in terms of valid move selection and overall gameplay quality.
- Despite improvements, neither agent dominated the other, highlighting the challenge of learning optimal strategies in this environment.
- The project also identified reproducibility and stability challenges when training RL agents in continuous, adversarial games.

## Future Work

- Investigate alternative RL algorithms or hybrid approaches to improve agent performance.
- Explore curriculum learning or reward shaping to accelerate learning.
- Analyze the impact of game mechanics and initial conditions on agent strategies and outcomes.
- Extend the framework to support human-vs-agent play and further evaluation. -->



## Resources
- [Rublisher website, including rules PDF and video](https://www.cmyk.games/products/lacuna)
- [Lacuna strategy discussion by game designer \[reddit\]](https://www.reddit.com/r/boardgames/comments/187cqiu/lacuna/)
- [SAC implementation paper](https://arxiv.org/abs/1812.05905)
- [PPO implementation paper](https://arxiv.org/abs/1707.06347)
