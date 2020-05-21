
# Ludo-AI

This project have been made by: **Emil Vincent Ancker**<br>
Student mail: **emanc16@student.sdu.dk**

## Methods
During this project multiple methods involving reinforcement learning, neural networks (NN) and genetic algorithms (GA) have been implemented and evaluated in order to develop an agent that plays Ludo at an acceptable level.

The following methods have been implemented:
* Q-Learning with an advanced state representation (3 bits for each token) allowing only one event to be active at once for each token.
* Q-Learning with a simple state representation (4 bits in total) allowing multiple events to be active at once.
* Genetic algorithms for learning optimal policy (using 4-bit binary encoded state-action representation)
* Genetic algorithms for evolution of weights used in a fully connected neural network.

## Win rates

Win rates against random player
| Advanced Q-learning | Simple Q-learning     | GA (4-bit state-action)    | GA evolving NN  |
| :-----------:|:------: |:-------------:| :-----:|
| 0.6567   | 0.8403  | 0.8118 | 0.9041 |

More results against several player types are available in the report.

## Ludo Game
The Ludo game that have been used during this project have been developed by Haukur Kristinsson and Rasmus Haugaard and can be found on the following Github repository: https://github.com/RasmusHaugaard/pyludo

The rules are *standard* Ludo rules, where a normal dice is used (faces from 1 to 6), the board contains the following special locations: globe (a safe location on which a token can not be send home by an opponent), star (landing on a star moves the player to the next star location), endgame zone (the last strip of locations up to goal, the endgame zone is considered safe and once it is entered it can not be left again unless a player sends one of its own tokens home by having three tokens on the same board location), goal (once a token lands on goal that token is considered out of the game, once all tokens lands on goal the game is finished).

The Ludo implementation in Python is able to simulate approximately 86 games per second with simple Ludo players, which naturally is being slowed down as the complexity of the players increase.
