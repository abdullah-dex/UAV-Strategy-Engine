# UAV Disaster-Zone Strategy Engine

## 1. Project Overview

The UAV Disaster-Zone Strategy Engine is an Artificial Intelligence project that uses a Genetic Algorithm (GA) to optimize flight plans for Unmanned Aerial Vehicles (UAVs) surveying a simulated disaster area.

The environment is represented as a two-dimensional grid. The system evolves candidate flight plans to improve the detection of high-priority disaster locations while considering estimated energy consumption, flight time, environmental hazards, and restricted areas.

A Random Search baseline is also implemented to provide a comparison with the Genetic Algorithm.

## 2. Objectives

The main objectives are:

- Represent UAV flight plans using chromosomes.
- Optimize waypoint coordinates, speed settings, and sensor power.
- Improve detection of high-priority disaster locations.
- Penalize excessive energy consumption and flight time.
- Penalize exposure to simulated environmental hazards.
- Penalize violations of no-fly zones and maximum route distance.
- Visualize fitness convergence, UAV routes, and sensor coverage.
- Compare the Genetic Algorithm against Random Search.
- Investigate the effects of population size and mutation rate.

## 3. Technologies Used

- Python 3
- NumPy
- Matplotlib
- Python standard libraries: `random`, `math`, and `csv`

## 4. Project Structure

The project contains the following files:

```text
UAV-Strategy-Engine/
├── main.py
├── main_backup.py
├── requirements.txt
├── README.md
├── fitness_convergence.png
├── best_uav_route.png
├── coverage_heatmap.png
├── baseline_comparison.png
└── sensitivity_results.csv
```

`main.py` contains the simulation, Genetic Algorithm, Random Search baseline, experiments, and visualization functions.

`main_backup.py` is a backup of an earlier prototype and is not required to run the improved simulation.

The PNG files contain generated visualizations. The CSV file contains the sensitivity experiment results.

## 5. Installation

Python 3 and `venv` are recommended.

### Step 1: Create a virtual environment

From the project directory, run:

```bash
python3 -m venv .venv
```

### Step 2: Activate the environment

On Ubuntu or another Linux system:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 6. Running the Simulation

Ensure the virtual environment is activated, then run:

```bash
python main.py
```

The program runs the Genetic Algorithm, evaluates a Random Search baseline, generates visualizations, and performs sensitivity experiments.

The program displays fitness values and information about the evolved UAV routes in the terminal.

Close each visualization window after viewing it so the program can continue.

## 7. Genetic Algorithm Design

### Chromosome Representation

Each chromosome represents a complete mission plan for the UAV fleet.

Each UAV plan contains:

- Waypoint coordinates `(x, y)`.
- Speed settings for each waypoint.
- Sensor power settings for each waypoint.

### Population Initialization

The initial population consists of randomly generated mission plans.

### Fitness Evaluation

The fitness function rewards weighted detection of high-priority targets and penalizes:

- Estimated energy consumption.
- Estimated flight time.
- Exposure to simulated hazards.
- No-fly-zone violations.
- Routes exceeding the maximum distance limit.

The objective weights are design choices used by the simulation and should be interpreted within the assumptions of this model.

### Selection

Tournament selection chooses parents from the current population.

### Crossover

Segment crossover exchanges portions of waypoint, speed, and sensor-power settings between parent chromosomes.

### Mutation

Mutation can modify waypoint coordinates, speed settings, and sensor power settings.

### Elitism

The best two chromosomes are preserved when creating the next generation.

## 8. Constraints and Assumptions

The environment uses a 50 × 50 grid with simulated disaster targets, hazard locations, and no-fly-zone cells.

The model uses an illustrative energy equation and estimated flight time. These values are not measurements from real UAV hardware.

No-fly-zone and maximum-distance violations are penalized by the fitness function. Penalty-based handling discourages violations but does not guarantee that every evolved route satisfies all constraints.

The sensor coverage heatmap estimates coverage around planned waypoints rather than simulating continuous sensor operation along every route segment.

## 9. Generated Outputs

The simulation generates the following files:

| File | Description |
|---|---|
| `fitness_convergence.png` | Best and average fitness across generations |
| `best_uav_route.png` | Evolved routes for the UAV fleet |
| `coverage_heatmap.png` | Estimated sensor coverage over the grid |
| `baseline_comparison.png` | Comparison of the GA's best fitness with Random Search scores |
| `sensitivity_results.csv` | Results for selected population-size and mutation-rate settings |

The results depend on the simulation design, objective weights, and random seed.

## 10. Experiments

The program compares the Genetic Algorithm with a Random Search baseline.

It also tests selected combinations of population size and mutation rate. The results are saved in `sensitivity_results.csv`.

These experiments provide data for examining the effect of selected algorithm parameters on solution quality. Further repeated trials and tests of selection pressure would provide a stronger sensitivity analysis.

## 11. Limitations

- The environment is simulated rather than based on live disaster data.
- The energy and flight-time calculations are simplified estimates.
- Constraint penalties do not guarantee feasible routes.
- Sensor coverage is approximated around waypoints.
- The experiments do not establish real-world UAV flight performance.
- Results should be interpreted within the assumptions of the simulation.

## 12. Learning Outcome

This project applies Artificial Intelligence techniques to a practical optimization problem. It represents mission plans as chromosomes, evaluates candidate solutions using a fitness function, and uses evolutionary search to improve UAV survey plans.

The project demonstrates how knowledge representation, evaluation, and optimization can be combined to support decision-making in a simulated disaster-response environment.