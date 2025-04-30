# Lacuna solver
> A project to create a machine learning agent to solve the board game
> [Lacuna](https://boardgamegeek.com/boardgame/386937/lacuna).
>
> Developed for 41118 Artificial Intelligence in Robotics - Autumn 2025

## Resources
- [Rublisher website, including rules PDF and video](https://www.cmyk.games/products/lacuna)
- [reddit strategy discussion from game designer](https://www.reddit.com/r/boardgames/comments/187cqiu/lacuna/)


## Instalation

Create a virtual enviroment with `python -m venv .venv`
Install all required packages with `pip install -r requirements.txt`

To activate the venv, which must be done before running code, run `source .venv/bin/activate`.
Deactivate with `deactivate`

List packages installed with `pip freeze -l` (local flag)

## Formats
<!-- TODO update -->
A "flower" is a token (note that the user placments aren't included) stored as a dictionary with the form **`{"pos": (x,y), "colour": color}`**
- `pos` is the position, as an `(x,y)` tupple
- `color` is an `int` (0 thru 6) representing which color the token is (we don't need to represent them as a string, `int` is easier)


## Plan:
RL Models:
- Q-Learning (Talita)
- PPO (Ben)

Rewards
- `+big` for winnng
- med for picking up flower
- small for near flowers
- negative for on token

Questions
- Action space will be massive
- Descretising vs continious?
- have you worked PPO?
- group count?