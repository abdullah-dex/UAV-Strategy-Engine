import random
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# UAV DISASTER-ZONE STRATEGY ENGINE
# Genetic Algorithm + Random Search Baseline
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

RANDOM_SEED = 42


# ============================================================
# ENVIRONMENT
# ============================================================

class UAVEnvironment:
    """Represents the disaster grid, targets, hazards, and no-fly zones."""

    def __init__(self):
        self.grid_size = GRID_SIZE

        # High-priority disaster locations: (x, y)
        self.targets = [
            (10, 10),
            (20, 35),
            (35, 15),
            (40, 40),
            (8, 38),
            (30, 8)
        ]

        # Simulated environmental hazard locations
        self.hazards = [
            (15, 20),
            (25, 25),
            (35, 30),
            (12, 32),
            (42, 18)
        ]

        # Restricted cells that UAVs must avoid
        self.no_fly_zones = [
            (20, 20),
            (21, 20),
            (20, 21),
            (21, 21),
            (30, 30),
            (31, 30),
            (30, 31),
            (31, 31)
        ]

        # Priority weights for disaster targets
        self.target_weights = {
            (10, 10): 3.0,
            (20, 35): 3.0,
            (35, 15): 2.0,
            (40, 40): 3.0,
            (8, 38): 2.0,
            (30, 8): 2.0
        }

    def in_bounds(self, point):
        x, y = point
        return (
            0 <= x < self.grid_size
            and 0 <= y < self.grid_size
        )

    def is_no_fly(self, point):
        return point in self.no_fly_zones

    def is_hazard(self, point):
        return point in self.hazards


# ============================================================
# CHROMOSOME REPRESENTATION
# ============================================================

class UAVPlan:
    """
    One UAV's plan.

    Each waypoint has:
    - x coordinate
    - y coordinate
    - speed setting
    - sensor power setting
    """

    def __init__(
        self,
        waypoints=None,
        speeds=None,
        sensor_powers=None
    ):
        if waypoints is None:
            self.waypoints = [
                (
                    random.randint(0, GRID_SIZE - 1),
                    random.randint(0, GRID_SIZE - 1)
                )
                for _ in range(NUM_WAYPOINTS)
            ]
        else:
            self.waypoints = list(waypoints)

        if speeds is None:
            self.speeds = [
                random.uniform(1.0, 5.0)
                for _ in range(NUM_WAYPOINTS)
            ]
        else:
            self.speeds = list(speeds)

        if sensor_powers is None:
            self.sensor_powers = [
                random.uniform(0.2, 1.0)
                for _ in range(NUM_WAYPOINTS)
            ]
        else:
            self.sensor_powers = list(sensor_powers)

    def copy(self):
        return UAVPlan(
            self.waypoints.copy(),
            self.speeds.copy(),
            self.sensor_powers.copy()
        )


class Chromosome:
    """A complete mission plan for the fleet of UAVs."""

    def __init__(self, plans=None):
        if plans is None:
            self.plans = [
                UAVPlan() for _ in range(NUM_UAVS)
            ]
        else:
            self.plans = plans

        self.fitness = None

    def copy(self):
        return Chromosome([
            plan.copy() for plan in self.plans
        ])


# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticAlgorithm:

    def __init__(
        self,
        environment,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        tournament_size=TOURNAMENT_SIZE
    ):
        self.environment = environment
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.population = [
            Chromosome() for _ in range(population_size)
        ]

        self.best_history = []
        self.average_history = []
        self.best = None

    @staticmethod
    def distance(a, b):
        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
        )

    def segment_hits_no_fly_zone(self, start, end):
        """
        Sample points along a route segment.
        Return True if the segment passes through a restricted cell.
        """
        distance = self.distance(start, end)
        steps = max(1, int(distance * 2))

        for step in range(steps + 1):
            fraction = step / steps

            x = round(
                start[0] + fraction * (end[0] - start[0])
            )
            y = round(
                start[1] + fraction * (end[1] - start[1])
            )

            if (x, y) in self.environment.no_fly_zones:
                return True

        return False

    def route_metrics(self, plan):
        """Calculate route distance, time, energy, and risk."""

        total_distance = 0.0
        total_time = 0.0
        total_energy = 0.0
        hazard_risk = 0.0
        violations = 0

        previous = (0, 0)

        for i, point in enumerate(plan.waypoints):
            speed = plan.speeds[i]
            power = plan.sensor_powers[i]

            segment_distance = self.distance(previous, point)
            total_distance += segment_distance

            # Higher speed reduces travel time but costs more energy.
            total_time += segment_distance / max(speed, 0.1)

            # Illustrative energy model:
            # movement energy + sensor energy
            total_energy += (
                segment_distance * (0.5 + 0.1 * speed)
                + power * 2.0
            )

            # Penalize restricted cells and segments crossing them.
            if self.segment_hits_no_fly_zone(previous, point):
                violations += 1

            # Penalize waypoints in hazardous areas.
            if point in self.environment.hazards:
                hazard_risk += 10.0

            previous = point

        # Check distance from the final waypoint back to the base.
        return_distance = self.distance(previous, (0, 0))
        total_distance += return_distance
        total_time += return_distance / max(plan.speeds[-1], 0.1)
        total_energy += (
            return_distance * (0.5 + 0.1 * plan.speeds[-1])
        )

        if total_distance > MAX_ROUTE_DISTANCE:
            violations += 1

        return {
            "distance": total_distance,
            "time": total_time,
            "energy": total_energy,
            "risk": hazard_risk,
            "violations": violations
        }

    def evaluate_fitness(self, chromosome):
        """
        Higher fitness is better.

        The score rewards weighted target detection and penalizes:
        energy, travel time, hazard exposure, and constraint violations.
        """

        targets = self.environment.targets
        target_coverage = {target: 0.0 for target in targets}

        total_energy = 0.0
        total_time = 0.0
        total_risk = 0.0
        total_violations = 0

        for plan in chromosome.plans:
            metrics = self.route_metrics(plan)

            total_energy += metrics["energy"]
            total_time += metrics["time"]
            total_risk += metrics["risk"]
            total_violations += metrics["violations"]

            # Detection probability depends on distance and sensor power.
            for i, point in enumerate(plan.waypoints):
                power = plan.sensor_powers[i]

                for target in targets:
                    d = self.distance(point, target)

                    if d <= SENSOR_RADIUS:
                        detection = power * max(
                            0.0, 1.0 - d / (SENSOR_RADIUS + 1)
                        )

                        # Combine repeated visits without exceeding 1.
                        old = target_coverage[target]
                        target_coverage[target] = (
                            1.0 - (1.0 - old) * (1.0 - detection)
                        )

        weighted_detection = sum(
            self.environment.target_weights[target]
            * target_coverage[target]
            for target in targets
        )

        # Explicit objective weights; these are design choices,
        # not values specified by the assignment brief.
        fitness = (
            20.0 * weighted_detection
            - 0.03 * total_energy
            - 0.10 * total_time
            - 1.0 * total_risk
            - 100.0 * total_violations
        )

        chromosome.fitness = fitness
        return fitness

    def select_parent(self):
        candidates = random.sample(
            self.population,
            min(self.tournament_size, len(self.population))
        )

        return max(
            candidates,
            key=lambda chromosome: chromosome.fitness
        )

    def crossover(self, parent1, parent2):
        """Segment crossover for waypoint, speed, and sensor genes."""

        child1 = parent1.copy()
        child2 = parent2.copy()

        for uav_index in range(NUM_UAVS):
            p1 = parent1.plans[uav_index]
            p2 = parent2.plans[uav_index]

            c1 = child1.plans[uav_index]
            c2 = child2.plans[uav_index]

            start = random.randint(0, NUM_WAYPOINTS - 2)
            end = random.randint(start + 1, NUM_WAYPOINTS)

            c1.waypoints[start:end] = p2.waypoints[start:end]
            c2.waypoints[start:end] = p1.waypoints[start:end]

            c1.speeds[start:end] = p2.speeds[start:end]
            c2.speeds[start:end] = p1.speeds[start:end]

            c1.sensor_powers[start:end] = (
                p2.sensor_powers[start:end]
            )
            c2.sensor_powers[start:end] = (
                p1.sensor_powers[start:end]
            )

        child1.fitness = None
        child2.fitness = None

        return child1, child2

    def mutate(self, chromosome):
        """Spatial mutation and speed/sensor adjustments."""

        for plan in chromosome.plans:
            for i in range(NUM_WAYPOINTS):

                if random.random() < self.mutation_rate:
                    x, y = plan.waypoints[i]

                    # Change position near the current waypoint.
                    x = max(
                        0,
                        min(GRID_SIZE - 1, x + random.randint(-5, 5))
                    )
                    y = max(
                        0,
                        min(GRID_SIZE - 1, y + random.randint(-5, 5))
                    )

                    plan.waypoints[i] = (x, y)

                if random.random() < self.mutation_rate:
                    plan.speeds[i] = max(
                        1.0,
                        min(
                            5.0,
                            plan.speeds[i] + random.uniform(-1.0, 1.0)
                        )
                    )

                if random.random() < self.mutation_rate:
                    plan.sensor_powers[i] = max(
                        0.2,
                        min(
                            1.0,
                            plan.sensor_powers[i]
                            + random.uniform(-0.2, 0.2)
                        )
                    )

        chromosome.fitness = None
        return chromosome

    def run(self, verbose=True):
        for generation in range(self.generations):

            for chromosome in self.population:
                self.evaluate_fitness(chromosome)

            self.population.sort(
                key=lambda chromosome: chromosome.fitness,
                reverse=True
            )

            best_score = self.population[0].fitness
            average_score = float(np.mean([
                chromosome.fitness
                for chromosome in self.population
            ]))

            self.best_history.append(best_score)
            self.average_history.append(average_score)

            if verbose:
                print(
                    f"Generation {generation + 1:03d}: "
                    f"Best Fitness = {best_score:.2f}, "
                    f"Average Fitness = {average_score:.2f}"
                )

            # Elitism: preserve the best two solutions.
            elite_count = min(2, self.population_size)

            new_population = [
                chromosome.copy()
                for chromosome in self.population[:elite_count]
            ]

            while len(new_population) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()

                child1, child2 = self.crossover(parent1, parent2)

                self.mutate(child1)
                self.mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        for chromosome in self.population:
            self.evaluate_fitness(chromosome)

        self.population.sort(
            key=lambda chromosome: chromosome.fitness,
            reverse=True
        )

        self.best = self.population[0].copy()
        self.best.fitness = self.population[0].fitness

        return self.best


# ============================================================
# RANDOM SEARCH BASELINE
# ============================================================

def random_search(environment, number_of_trials=200):
    """Evaluate randomly generated mission plans."""

    evaluator = GeneticAlgorithm(
        environment,
        population_size=2,
        generations=1
    )

    best = None
    scores = []

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

    plt.plot(
        algorithm.best_history,
        label="Best Fitness"
    )
    plt.plot(
        algorithm.average_history,
        label="Average Fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Fitness Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_convergence.png", dpi=150)
    plt.show()
    plt.close()


def plot_routes(chromosome, environment):
    plt.figure(figsize=(9, 8))

    tx, ty = zip(*environment.targets)
    hx, hy = zip(*environment.hazards)
    nx, ny = zip(*environment.no_fly_zones)

    plt.scatter(
        tx, ty, marker="*", s=180, label="Priority Targets"
    )
    plt.scatter(
        hx, hy, marker="x", s=100, label="Hazards"
    )
    plt.scatter(
        nx, ny, marker="s", s=90, label="No-Fly Zones"
    )

    for index, plan in enumerate(chromosome.plans):
        route = [(0, 0)] + plan.waypoints + [(0, 0)]

        xs = [point[0] for point in route]
        ys = [point[1] for point in route]

        plt.plot(
            xs,
            ys,
            marker="o",
            label=f"UAV {index + 1}"
        )

    plt.xlim(0, GRID_SIZE - 1)
    plt.ylim(0, GRID_SIZE - 1)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Evolved UAV Fleet Flight Routes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("best_uav_route.png", dpi=150)
    plt.show()
    plt.close()


def plot_coverage(chromosome, environment):
    """
    Create an approximate sensor coverage heatmap.
    The heatmap represents coverage near the planned waypoints.
    """

    coverage = np.zeros(
        (GRID_SIZE, GRID_SIZE),
        dtype=float
    )

    for plan in chromosome.plans:
        for i, (x, y) in enumerate(plan.waypoints):
            power = plan.sensor_powers[i]

            for gx in range(GRID_SIZE):
                for gy in range(GRID_SIZE):
                    d = math.sqrt(
                        (gx - x) ** 2 + (gy - y) ** 2
                    )

                    if d <= SENSOR_RADIUS:
                        strength = power * max(
                            0.0,
                            1.0 - d / (SENSOR_RADIUS + 1)
                        )

                        # Combine overlapping sensor coverage.
                        coverage[gx, gy] = max(
                            coverage[gx, gy],
                            strength
                        )

    plt.figure(figsize=(9, 8))

    plt.imshow(
        coverage.T,
        origin="lower",
        extent=[0, GRID_SIZE - 1, 0, GRID_SIZE - 1],
        cmap="hot",
        interpolation="nearest"
    )

    tx, ty = zip(*environment.targets)
    nx, ny = zip(*environment.no_fly_zones)

    plt.scatter(
        tx, ty, marker="*", s=150, label="Priority Targets"
    )
    plt.scatter(
        nx, ny, marker="s", s=70, label="No-Fly Zones"
    )

    plt.colorbar(label="Estimated Sensor Coverage")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("UAV Sensor Coverage Heatmap")
    plt.legend()
    plt.tight_layout()
    plt.savefig("coverage_heatmap.png", dpi=150)
    plt.show()
    plt.close()


def plot_baseline_comparison(ga_score, random_scores):
    plt.figure(figsize=(8, 5))

    plt.boxplot(
        random_scores,
        tick_labels=["Random Search"]
    )

    plt.scatter(
        [1],
        [ga_score],
        marker="D",
        s=80,
        label="Genetic Algorithm Best"
    )

    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm vs Random Search")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=150)
    plt.show()
    plt.close()


# ============================================================
# EXPERIMENTS
# ============================================================

def run_sensitivity_experiments(environment):
    """
    Compare selected population sizes and mutation rates.
    Results are written to sensitivity_results.csv.
    """

    settings = [
        (20, 0.10),
        (40, 0.10),
        (40, 0.25),
        (60, 0.10)
    ]

    results = []

    for population_size, mutation_rate in settings:
        print(
            "\nSensitivity experiment:",
            f"population={population_size},",
            f"mutation={mutation_rate}"
        )

        experiment = GeneticAlgorithm(
            environment,
            population_size=population_size,
            generations=25,
            mutation_rate=mutation_rate
        )

        best = experiment.run(verbose=False)

        results.append({
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "best_fitness": best.fitness
        })

        print("Best fitness:", round(best.fitness, 2))

    with open(
        "sensitivity_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "population_size",
                "mutation_rate",
                "best_fitness"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved sensitivity_results.csv")


# ============================================================
# MAIN PROGRAM
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
    best_chromosome = algorithm.run()

    print("\nGenetic Algorithm completed.")
    print("Best fitness:", round(best_chromosome.fitness, 2))

    for index, plan in enumerate(best_chromosome.plans):
        print(f"\nUAV {index + 1} route:")
        print(plan.waypoints)
        print("Speeds:", [round(value, 2) for value in plan.speeds])
        print(
            "Sensor powers:",
            [round(value, 2) for value in plan.sensor_powers]
        )

        metrics = algorithm.route_metrics(plan)

        print("Route distance:", round(metrics["distance"], 2))
        print("Estimated flight time:", round(metrics["time"], 2))
        print("Estimated energy:", round(metrics["energy"], 2))
        print("Hazard risk:", round(metrics["risk"], 2))
        print("Constraint violations:", metrics["violations"])

    print("\nRunning random-search baseline...")

    random_best, random_scores = random_search(
        environment,
        number_of_trials=200
    )

    print("Best random-search fitness:", round(random_best.fitness, 2))
    print(
        "Average random-search fitness:",
        round(float(np.mean(random_scores)), 2)
    )

    print("\nGenerating visualizations...")

    plot_convergence(algorithm)
    plot_routes(best_chromosome, environment)
    plot_coverage(best_chromosome, environment)
    plot_baseline_comparison(
        best_chromosome.fitness,
        random_scores
    )

    print("\nRunning sensitivity experiments...")
    run_sensitivity_experiments(environment)

    print("\n" + "=" * 60)
    print("SIMULATION FINISHED")
    print("Check the project folder for PNG images and CSV results.")
    print("=" * 60)


if __name__ == "__main__":
    main()