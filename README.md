# Lacuna solver
> A project to create machine learning agents to solve the board game
> [Lacuna](https://boardgamegeek.com/boardgame/386937/lacuna).
>
> Developed for 41118 Artificial Intelligence
> in Robotics \- Autumn 2025.
> Made by Talita James and Benjamin Cooper, as group 4.




## Installation
To run this project, we use a python virtual enviroment (venv).
```bash
python -m venv .venv # create the venv
source .venv/bin/activate # startup the venv
pip install -r requirements.txt # install the required packages
mkdir out # creates location for checkpoint saves
```
Starting the venv must be done before runing code, in each new window.
Use `source .venv/bin/activate` in this directory,
and deactivate with `deactivate` from any dir

## Running
To run the code

### Args
To change the file execution, there are some arguments that may be passed.
Detailed below, with flag abbreviations annotated in **bold**.

| Command        |  Data type & Default Value | Description |
|---------------:|--------------|--------------------|
| -**-v**erbose  | `bool False` | Controls the amount of information sent to stdout             |
| -**-s**how     | `bool False` | Should there be a visual representation of the board after each turn? |
| -**-l**oad     | `bool False` | Existing              |
| -**-e**pisodes | `int 10_000` | The number of episodes (games) to train the models on |

## Resources
- [Rublisher website, including rules PDF and video](https://www.cmyk.games/products/lacuna)
- [Lacuna strategy discussion by game designer \[reddit\]](https://www.reddit.com/r/boardgames/comments/187cqiu/lacuna/)
- [SAC implementation paper](https://arxiv.org/abs/1812.05905)