# UAV Disaster-Zone Strategy Engine

## Overview

The UAV Disaster-Zone Strategy Engine is an Artificial Intelligence project that uses a Genetic Algorithm (GA) to optimize routes for two Unmanned Aerial Vehicles (UAVs) operating in a simulated disaster environment.

The system aims to balance target coverage, energy consumption, mission time, and hazard exposure while accounting for route constraints.

## Features

- Genetic Algorithm for multi-objective route optimization.
- Chromosome representation of UAV waypoints, speeds, and sensor power.
- Tournament selection, crossover, mutation, and elitism.
- Fitness evaluation using target priorities and operational penalties.
- No-fly-zone and hazard-risk evaluation.
- Population diversity monitoring and random-candidate injection when diversity becomes low.
- Random-search baseline for comparison.
- Sensitivity experiments examining population size, mutation rate, and tournament size.
- Visualizations of routes, target coverage, fitness convergence, population diversity, and baseline performance.
- CSV output containing sensitivity experiment results.

## Requirements

- Python 3
- Packages listed in `requirements.txt`

## Installation

Clone the repository:

```bash
git clone https://github.com/abdullah-dex/UAV-Strategy-Engine.git
cd UAV-Strategy-Engine
```

Create and activate a virtual environment on Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

Run the main program:

```bash
python main.py
```

The program executes the Genetic Algorithm, evaluates a random-search baseline, generates visualizations, and performs sensitivity experiments.

The simulation may take some time to complete.

## Output Files

Generated results are saved in the `outputs/` directory:

| File | Description |
|---|---|
| `best_uav_route.png` | Visual representation of the selected UAV routes |
| `coverage_heatmap.png` | Estimated sensor coverage across the environment |
| `fitness_convergence.png` | Fitness progress across GA generations |
| `population_diversity.png` | Population diversity during optimization |
| `baseline_comparison.png` | Comparison visualization for the random-search baseline |
| `sensitivity_results.csv` | Results from the hyperparameter sensitivity experiments |

## Algorithm Configuration

The simulation uses two UAVs, each represented by a sequence of eight waypoints. The chromosome also encodes speed and sensor-power settings.

The fitness function evaluates target detection and penalizes energy use, mission time, hazard risk, and constraint violations. Genetic operators generate and improve candidate mission plans over successive generations.

## Experimental Results

In one recorded simulation run:

- Best Genetic Algorithm fitness: **239.11**
- Best random-search fitness: **-73.21**
- Average random-search fitness: **-360.80**
- UAV 1 reported constraint violations: **0**
- UAV 2 reported constraint violations: **0**

Six sensitivity experiments were conducted using different population sizes, mutation rates, and tournament sizes.

These results describe the recorded simulation run. Because the algorithm uses randomness, results can vary between runs.

## Project Structure

```text
UAV-Strategy-Engine/
├── main.py
├── main_previous.py
├── README.md
├── requirements.txt
├── outputs/
│   ├── baseline_comparison.png
│   ├── best_uav_route.png
│   ├── coverage_heatmap.png
│   ├── fitness_convergence.png
│   ├── population_diversity.png
│   └── sensitivity_results.csv
└── report/
    └── technical_report.docx
```

The exact files present may change as the project is updated.

## Limitations

This project is a simulation for academic purposes. Its environment, fitness weights, hazard model, and operational calculations use simplified assumptions. The results should not be interpreted as certification of real-world UAV safety.

## Purpose

This project demonstrates the application of Genetic Algorithms, optimization, constraint handling, and experimental evaluation to an AI-based route-planning problem.