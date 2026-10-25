import random
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# UAV DISASTER-ZONE STRATEGY ENGINE
# Genetic Algorithm + Random-Search Baseline
# ============================================================

GRID_SIZE = 50
NUM_UAVS = 2
NUM_WAYPOINTS = 8

POPULATION_SIZE = 40
GENERATIONS = 50
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 3

MAX_ROUTE_DISTANCE = 200.0
SENSOR_RADIUS = 4.0
HAZARD_INFLUENCE_RADIUS = 5.0
RANDOM_SEED = 42

# All outputs are written to the outputs folder beside this script.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT
# ============================================================

class UAVEnvironment:
    """Synthetic disaster grid, targets, hazards, and restricted cells."""

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.targets = [
            (10, 10), (20, 35), (35, 15),
            (40, 40), (8, 38), (30, 8)
        ]
        self.hazards = [
            (15, 20), (25, 25), (35, 30),
            (12, 32), (42, 18)
        ]
        self.no_fly_zones = [
            (20, 20), (21, 20), (20, 21), (21, 21),
            (30, 30), (31, 30), (30, 31), (31, 31)
        ]
        self.target_weights = {
            (10, 10): 3.0, (20, 35): 3.0, (35, 15): 2.0,
            (40, 40): 3.0, (8, 38): 2.0, (30, 8): 2.0
        }

    def in_bounds(self, point):
        x, y = point
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size


# ============================================================
# CHROMOSOME REPRESENTATION
# ============================================================

class UAVPlan:
    """One UAV route with waypoint coordinates, speed, and sensor power."""

    def __init__(self, waypoints=None, speeds=None, sensor_powers=None):
        self.waypoints = list(waypoints) if waypoints is not None else [
            (random.randint(0, GRID_SIZE - 1),
             random.randint(0, GRID_SIZE - 1))
            for _ in range(NUM_WAYPOINTS)
        ]
        self.speeds = list(speeds) if speeds is not None else [
            random.uniform(1.0, 5.0) for _ in range(NUM_WAYPOINTS)
        ]
        self.sensor_powers = list(sensor_powers) if sensor_powers is not None else [
            random.uniform(0.2, 1.0) for _ in range(NUM_WAYPOINTS)
        ]

    def copy(self):
        return UAVPlan(self.waypoints.copy(), self.speeds.copy(),
                       self.sensor_powers.copy())


class Chromosome:
    """A complete fleet mission plan."""

    def __init__(self, plans=None):
        self.plans = plans if plans is not None else [
            UAVPlan() for _ in range(NUM_UAVS)
        ]
        self.fitness = None

    def copy(self):
        child = Chromosome([plan.copy() for plan in self.plans])
        child.fitness = self.fitness
        return child


# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticAlgorithm:
    def __init__(self, environment, population_size=POPULATION_SIZE,
                 generations=GENERATIONS, mutation_rate=MUTATION_RATE,
                 tournament_size=TOURNAMENT_SIZE, diversity_threshold=0.08):
        self.environment = environment
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.diversity_threshold = diversity_threshold
        self.population = [Chromosome() for _ in range(population_size)]
        self.best_history = []
        self.average_history = []
        self.diversity_history = []
        self.best = None

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def segment_hits_no_fly_zone(self, start, end):
        """Dense grid sampling to approximate segment/restricted-cell intersection."""
        distance = self.distance(start, end)
        steps = max(1, int(distance * 4))
        for step in range(steps + 1):
            fraction = step / steps
            x = round(start[0] + fraction * (end[0] - start[0]))
            y = round(start[1] + fraction * (end[1] - start[1]))
            if (x, y) in self.environment.no_fly_zones:
                return True
        return False

    def route_metrics(self, plan):
        total_distance = total_time = total_energy = 0.0
        hazard_risk = 0.0
        violations = 0
        previous = (0, 0)

        for i, point in enumerate(plan.waypoints):
            speed = plan.speeds[i]
            power = plan.sensor_powers[i]
            segment_distance = self.distance(previous, point)
            total_distance += segment_distance
            total_time += segment_distance / max(speed, 0.1)
            total_energy += segment_distance * (0.5 + 0.1 * speed) + power * 2.0

            if self.segment_hits_no_fly_zone(previous, point):
                violations += 1

            # Risk rises near hazards, not only on exact hazard coordinates.
            for hazard in self.environment.hazards:
                d = self.distance(point, hazard)
                if d < HAZARD_INFLUENCE_RADIUS:
                    hazard_risk += (HAZARD_INFLUENCE_RADIUS - d) / HAZARD_INFLUENCE_RADIUS

            previous = point

        return_distance = self.distance(previous, (0, 0))
        total_distance += return_distance
        total_time += return_distance / max(plan.speeds[-1], 0.1)
        total_energy += return_distance * (0.5 + 0.1 * plan.speeds[-1])

        if total_distance > MAX_ROUTE_DISTANCE:
            violations += 1

        return {"distance": total_distance, "time": total_time,
                "energy": total_energy, "risk": hazard_risk,
                "violations": violations}

    def evaluate_fitness(self, chromosome):
        target_coverage = {target: 0.0 for target in self.environment.targets}
        total_energy = total_time = total_risk = 0.0
        total_violations = 0

        for plan in chromosome.plans:
            metrics = self.route_metrics(plan)
            total_energy += metrics["energy"]
            total_time += metrics["time"]
            total_risk += metrics["risk"]
            total_violations += metrics["violations"]

            for i, point in enumerate(plan.waypoints):
                power = plan.sensor_powers[i]
                for target in self.environment.targets:
                    d = self.distance(point, target)
                    if d <= SENSOR_RADIUS:
                        detection = power * max(0.0, 1.0 - d / (SENSOR_RADIUS + 1))
                        old = target_coverage[target]
                        target_coverage[target] = 1.0 - (1.0 - old) * (1.0 - detection)

        weighted_detection = sum(
            self.environment.target_weights[t] * target_coverage[t]
            for t in self.environment.targets
        )
        # These coefficients are simulation design choices, not real-world constants.
        fitness = (20.0 * weighted_detection - 0.03 * total_energy
                   - 0.10 * total_time - 1.0 * total_risk
                   - 100.0 * total_violations)
        chromosome.fitness = fitness
        return fitness

    def select_parent(self):
        candidates = random.sample(
            self.population, min(self.tournament_size, len(self.population))
        )
        return max(candidates, key=lambda c: c.fitness)

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        for uav_index in range(NUM_UAVS):
            p1, p2 = parent1.plans[uav_index], parent2.plans[uav_index]
            c1, c2 = child1.plans[uav_index], child2.plans[uav_index]
            start = random.randint(0, NUM_WAYPOINTS - 2)
            end = random.randint(start + 1, NUM_WAYPOINTS)
            for attr in ("waypoints", "speeds", "sensor_powers"):
                getattr(c1, attr)[start:end] = getattr(p2, attr)[start:end]
                getattr(c2, attr)[start:end] = getattr(p1, attr)[start:end]
        child1.fitness = child2.fitness = None
        return child1, child2

    def mutate(self, chromosome):
        for plan in chromosome.plans:
            for i in range(NUM_WAYPOINTS):
                if random.random() < self.mutation_rate:
                    x, y = plan.waypoints[i]
                    x = max(0, min(GRID_SIZE - 1, x + random.randint(-5, 5)))
                    y = max(0, min(GRID_SIZE - 1, y + random.randint(-5, 5)))
                    plan.waypoints[i] = (x, y)
                if random.random() < self.mutation_rate:
                    plan.speeds[i] = max(1.0, min(5.0, plan.speeds[i] + random.uniform(-1, 1)))
                if random.random() < self.mutation_rate:
                    plan.sensor_powers[i] = max(
                        0.2, min(1.0, plan.sensor_powers[i] + random.uniform(-0.2, 0.2))
                    )
        chromosome.fitness = None
        return chromosome

    def population_diversity(self):
        """Mean normalized waypoint difference from the population centroid."""
        if not self.population:
            return 0.0
        values = []
        for chromosome in self.population:
            coords = [v for plan in chromosome.plans for point in plan.waypoints for v in point]
            values.append(coords)
        matrix = np.asarray(values, dtype=float)
        if len(matrix) < 2:
            return 0.0
        # Coordinate standard deviation normalized by the maximum coordinate span.
        return float(np.mean(np.std(matrix, axis=0)) / max(1.0, GRID_SIZE - 1))

    def run(self, verbose=True):
        for generation in range(self.generations):
            for chromosome in self.population:
                if chromosome.fitness is None:
                    self.evaluate_fitness(chromosome)
            self.population.sort(key=lambda c: c.fitness, reverse=True)
            scores = [c.fitness for c in self.population]
            best_score = scores[0]
            avg_score = float(np.mean(scores))
            diversity = self.population_diversity()
            self.best_history.append(best_score)
            self.average_history.append(avg_score)
            self.diversity_history.append(diversity)

            if verbose:
                print(f"Generation {generation + 1:03d}: Best Fitness = {best_score:.2f}, "
                      f"Average Fitness = {avg_score:.2f}, Diversity = {diversity:.3f}")

            elite_count = min(2, self.population_size)
            new_population = [c.copy() for c in self.population[:elite_count]]

            # If diversity is low, reserve some slots for fresh random candidates.
            inject_count = 0
            if diversity < self.diversity_threshold and self.population_size > elite_count:
                inject_count = max(1, self.population_size // 10)

            while len(new_population) < self.population_size - inject_count:
                p1, p2 = self.select_parent(), self.select_parent()
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size - inject_count:
                    new_population.append(c2)

            for _ in range(inject_count):
                if len(new_population) < self.population_size:
                    new_population.append(Chromosome())
            self.population = new_population

        for chromosome in self.population:
            if chromosome.fitness is None:
                self.evaluate_fitness(chromosome)
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        self.best = self.population[0].copy()
        return self.best


# ============================================================
# RANDOM SEARCH BASELINE
# ============================================================

def random_search(environment, number_of_trials=200):
    evaluator = GeneticAlgorithm(environment, population_size=2, generations=1)
    best, scores = None, []
    for _ in range(number_of_trials):
        candidate = Chromosome()
        score = evaluator.evaluate_fitness(candidate)
        scores.append(score)
        if best is None or score > best.fitness:
            best = candidate.copy()
            best.fitness = score
    return best, scores


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_convergence(algorithm):
    plt.figure(figsize=(9, 5))
    plt.plot(algorithm.best_history, label="Best Fitness")
    plt.plot(algorithm.average_history, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Fitness Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fitness_convergence.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(algorithm.diversity_history, label="Normalized Population Diversity")
    plt.axhline(algorithm.diversity_threshold, linestyle="--", label="Injection Threshold")
    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.title("Population Diversity Across Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "population_diversity.png", dpi=150)
    plt.close()


def plot_routes(chromosome, environment):
    plt.figure(figsize=(9, 8))
    tx, ty = zip(*environment.targets)
    hx, hy = zip(*environment.hazards)
    nx, ny = zip(*environment.no_fly_zones)
    plt.scatter(tx, ty, marker="*", s=180, label="Priority Targets")
    plt.scatter(hx, hy, marker="x", s=100, label="Hazards")
    plt.scatter(nx, ny, marker="s", s=90, label="No-Fly Cells")
    for index, plan in enumerate(chromosome.plans):
        route = [(0, 0)] + plan.waypoints + [(0, 0)]
        plt.plot([p[0] for p in route], [p[1] for p in route],
                 marker="o", label=f"UAV {index + 1}")
    plt.xlim(0, GRID_SIZE - 1)
    plt.ylim(0, GRID_SIZE - 1)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Evolved UAV Fleet Flight Routes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_uav_route.png", dpi=150)
    plt.close()


def plot_coverage(chromosome, environment):
    coverage = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for plan in chromosome.plans:
        for i, (x, y) in enumerate(plan.waypoints):
            power = plan.sensor_powers[i]
            for gx in range(GRID_SIZE):
                for gy in range(GRID_SIZE):
                    d = math.hypot(gx - x, gy - y)
                    if d <= SENSOR_RADIUS:
                        strength = power * max(0.0, 1.0 - d / (SENSOR_RADIUS + 1))
                        coverage[gx, gy] = max(coverage[gx, gy], strength)
    plt.figure(figsize=(9, 8))
    plt.imshow(coverage.T, origin="lower",
               extent=[0, GRID_SIZE - 1, 0, GRID_SIZE - 1],
               cmap="hot", interpolation="nearest")
    tx, ty = zip(*environment.targets)
    nx, ny = zip(*environment.no_fly_zones)
    plt.scatter(tx, ty, marker="*", s=150, label="Priority Targets")
    plt.scatter(nx, ny, marker="s", s=70, label="No-Fly Cells")
    plt.colorbar(label="Estimated Sensor Coverage")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Final Solution Sensor Coverage Heatmap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coverage_heatmap.png", dpi=150)
    plt.close()


def plot_baseline_comparison(ga_score, random_scores):
    plt.figure(figsize=(8, 5))
    plt.boxplot(random_scores, tick_labels=["Random Search"])
    plt.scatter([1], [ga_score], marker="D", s=80, label="GA Best")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm vs Random Search")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=150)
    plt.close()


# ============================================================
# EXPERIMENTS
# ============================================================

def run_sensitivity_experiments(environment):
    # Vary one parameter at a time where possible.
    settings = [
        {"population_size": 20, "mutation_rate": 0.10, "tournament_size": 3},
        {"population_size": 40, "mutation_rate": 0.10, "tournament_size": 3},
        {"population_size": 60, "mutation_rate": 0.10, "tournament_size": 3},
        {"population_size": 40, "mutation_rate": 0.25, "tournament_size": 3},
        {"population_size": 40, "mutation_rate": 0.10, "tournament_size": 2},
        {"population_size": 40, "mutation_rate": 0.10, "tournament_size": 5},
    ]
    results = []
    for index, setting in enumerate(settings, start=1):
        print(f"Sensitivity {index}/{len(settings)}: {setting}")
        random.seed(RANDOM_SEED + 100 * index)
        np.random.seed(RANDOM_SEED + 100 * index)
        experiment = GeneticAlgorithm(
            environment,
            population_size=setting["population_size"],
            generations=25,
            mutation_rate=setting["mutation_rate"],
            tournament_size=setting["tournament_size"]
        )
        best = experiment.run(verbose=False)
        results.append({
            **setting,
            "generations": 25,
            "best_fitness": round(best.fitness, 4),
            "final_average_fitness": round(experiment.average_history[-1], 4),
            "final_diversity": round(experiment.diversity_history[-1], 4)
        })
        print("  best fitness:", round(best.fitness, 2))

    output = OUTPUT_DIR / "sensitivity_results.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {output}")


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    environment = UAVEnvironment()

    print("=" * 60)
    print("UAV DISASTER-ZONE STRATEGY ENGINE")
    print("=" * 60)
    print("\nRunning Genetic Algorithm...")
    algorithm = GeneticAlgorithm(environment)
    best = algorithm.run()

    print("\nGenetic Algorithm completed.")
    print("Best fitness:", round(best.fitness, 2))
    for index, plan in enumerate(best.plans):
        print(f"\nUAV {index + 1} waypoints:", plan.waypoints)
        print("Speeds:", [round(v, 2) for v in plan.speeds])
        print("Sensor powers:", [round(v, 2) for v in plan.sensor_powers])
        metrics = algorithm.route_metrics(plan)
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {round(value, 2) if isinstance(value, float) else value}")

    print("\nRunning random-search baseline...")
    random_best, random_scores = random_search(environment, number_of_trials=200)
    print("Best random-search fitness:", round(random_best.fitness, 2))
    print("Average random-search fitness:", round(float(np.mean(random_scores)), 2))

    print("\nGenerating visualizations...")
    plot_convergence(algorithm)
    plot_routes(best, environment)
    plot_coverage(best, environment)
    plot_baseline_comparison(best.fitness, random_scores)

    print("\nRunning sensitivity experiments...")
    run_sensitivity_experiments(environment)

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(" -", file.name)
    print("\nSIMULATION FINISHED")


if __name__ == "__main__":
    main()
