# Wumpus-World
This project involves developing a rational agent to excel in the Wumpus World game using model checking and reinforcement learning techniques. The agent navigates the environment, avoids hazards, and aims to find gold, demonstrating advanced decision-making capabilities.

## Key Features

- **Model Checking**: Utilizes truth-table enumeration to identify safe moves based on percepts.
- **Probabilistic Model Checking**: Enhances decision-making under uncertainty by calculating the probability of safe moves.
- **Reinforcement Learning**: Implements SARSA (State-Action-Reward-State-Action) temporal difference learning to adapt strategies based on environmental interactions.

## Methodology

### Version 1: Model Checking

- **Map Maintenance**: Maintains a perceptual map of the Wumpus world, updating it with sensory data.
- **Model Construction and Deduction**: Constructs models based on percepts and validates them to determine safe moves.
- **Decision Making**: Chooses moves based on validated models to ensure safety.

### Version 2: Probabilistic Model Checking

- **Integration of Probabilities**: Calculates the likelihood of each state leading to a safe outcome.
- **Model Evaluation**: Constructs and evaluates models to determine the probability of safety for each move.
- **Decision Making**: Prioritizes moves with higher safety probabilities.

### Version 3: Reinforcement Learning

- **Q-table Initialization**: Stores value estimates for state-action pairs.
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation in decision-making.
- **SARSA Update Rule**: Continuously updates the Q-table based on rewards received and the estimated value of subsequent state-action pairs.

## Experimental Setup and Results

### Agents Compared

- **Random Agent**: Makes decisions based solely on random choice.
- **Probabilistic Model Checking Agent**: Utilizes systematic decision-making based on model checking.
- **Reinforcement Learning Agent**: Uses SARSA to adapt strategies based on past experiences.

### Key Metrics Collected

- **Rewards per Episode**: Total rewards accumulated during each simulation.
- **Steps per Episode**: Total number of moves taken to complete an episode.
- **Success**: Number of times the agent successfully retrieved the gold.

### Results

- **Random Agent**: Average Reward: -818.1, Average Steps: 18.2, Average Success: 0.1
- **Probabilistic Model Checking Agent**: Average Reward: 89.0, Average Steps: 11.6, Average Success: 0.6
- **Reinforcement Learning Agent**: Average Reward: -152.52, Average Steps: 12.98, Average Success: 0.46

## How to Run the Project

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/kaylalaufer/Wumpus-World.git
   cd Wumpus-World
