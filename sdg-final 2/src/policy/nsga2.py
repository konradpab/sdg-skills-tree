"""
NSGA-II Multi-Objective Optimization for Intervention Selection.

Generates Pareto-optimal intervention packages that trade off
multiple objectives (SDG progress, CO₂ reduction, feasibility, cost)
without requiring pre-specified weights.

This complements the weighted-sum planner (planner.py) by providing:
  - A Pareto front of non-dominated solutions
  - Visualization of trade-offs between objectives
  - Comparison with the weighted-sum solution

Reference: Deb et al. (2002), "A Fast and Elitist Multiobjective
Genetic Algorithm: NSGA-II", IEEE Trans. Evol. Computation, 6(2).
"""

import numpy as np
import json
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


def _evaluate_package(selected_mask, interventions, country_state, prerequisites_map):
    """
    Evaluate a candidate intervention package on 4 objectives.
    Returns (sdg_score, co2_reduction, feasibility, neg_cost).
    All objectives are MAXIMIZED.
    """
    selected = [interventions[i] for i, s in enumerate(selected_mask) if s]

    if not selected:
        return (0.0, 0.0, 0.0, 0.0)

    # Check prerequisite violations
    selected_ids = {intv["tech_id"] for intv in selected}
    prereq_violations = 0
    for intv in selected:
        for prereq in intv.get("prerequisites_hard", []):
            if prereq not in selected_ids:
                prereq_violations += 1

    if prereq_violations > 0:
        # Heavy penalty for prerequisite violations
        penalty = 0.3 * prereq_violations
        return (-penalty, -penalty, -penalty, -penalty)

    # Objective 1: SDG improvement (sum of positive effects)
    sdg = 0
    for intv in selected:
        eff = intv.get("estimated_effects", {})
        sdg += eff.get("renewables_share_pct", 0) * 0.03
        sdg += eff.get("electricity_access_pct", 0) * 0.03
        sdg += eff.get("energy_intensity_mj_usd", 0) * (-0.04)
        sdg += eff.get("rd_expenditure_gdp_pct", 0) * 0.05

    # Objective 2: CO₂ reduction (negative emissions change is good)
    co2_reduction = 0
    for intv in selected:
        co2_reduction -= intv.get("estimated_effects", {}).get("co2_per_capita_t", 0)

    # Objective 3: Average feasibility
    from src.policy.planner import check_prerequisites
    feas_scores = []
    for intv in selected:
        _, penalty, _ = check_prerequisites(intv, country_state)
        feas_scores.append(max(0, 1 - penalty))
    feasibility = np.mean(feas_scores) if feas_scores else 0

    # Objective 4: Negative cost (lower cost = higher score)
    cost_map = {"low": 0.9, "medium": 0.5, "high": 0.2}
    neg_cost = np.mean([cost_map.get(intv["cost_category"], 0.5) for intv in selected])

    return (sdg, co2_reduction, feasibility, neg_cost)


def _dominates(obj_a, obj_b):
    """True if solution a Pareto-dominates solution b (all >= and at least one >)."""
    at_least_one_better = False
    for a, b in zip(obj_a, obj_b):
        if a < b:
            return False
        if a > b:
            at_least_one_better = True
    return at_least_one_better


def _fast_non_dominated_sort(objectives):
    """NSGA-II fast non-dominated sorting. Returns list of fronts (index lists)."""
    n = len(objectives)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated_set[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # remove empty last front


def _crowding_distance(objectives, front):
    """Compute crowding distance for solutions in a front."""
    n = len(front)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n
    n_obj = len(objectives[front[0]])

    for m in range(n_obj):
        sorted_indices = sorted(range(n), key=lambda i: objectives[front[i]][m])
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        obj_range = objectives[front[sorted_indices[-1]]][m] - objectives[front[sorted_indices[0]]][m]
        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                objectives[front[sorted_indices[i + 1]]][m]
                - objectives[front[sorted_indices[i - 1]]][m]
            ) / obj_range

    return distances


def nsga2_optimize(interventions, country_state, pop_size=100, n_generations=50,
                   max_package_size=8, crossover_rate=0.8, mutation_rate=0.15):
    """
    Run NSGA-II to find Pareto-optimal intervention packages.

    Each individual is a binary vector (1 = include intervention, 0 = exclude).
    Objectives: maximize (SDG, CO₂ reduction, feasibility, -cost).

    Returns list of Pareto-optimal solutions with objectives and packages.
    """
    n_interventions = len(interventions)
    prereqs = {intv["tech_id"]: intv.get("prerequisites_hard", []) for intv in interventions}

    # Initialize population: random binary vectors
    population = []
    for _ in range(pop_size):
        n_selected = np.random.randint(2, min(max_package_size + 1, n_interventions + 1))
        individual = np.zeros(n_interventions, dtype=int)
        chosen = np.random.choice(n_interventions, size=n_selected, replace=False)
        individual[chosen] = 1
        population.append(individual)

    for gen in range(n_generations):
        # Evaluate all individuals
        objectives = [_evaluate_package(ind, interventions, country_state, prereqs)
                      for ind in population]

        # Non-dominated sorting
        fronts = _fast_non_dominated_sort(objectives)

        # Select next generation using fronts + crowding distance
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend([population[i] for i in front])
            else:
                # Use crowding distance to break ties
                cd = _crowding_distance(objectives, front)
                sorted_by_cd = sorted(zip(front, cd), key=lambda x: -x[1])
                remaining = pop_size - len(next_pop)
                next_pop.extend([population[idx] for idx, _ in sorted_by_cd[:remaining]])
                break

        # Generate offspring via crossover and mutation
        offspring = []
        while len(offspring) < pop_size:
            if np.random.random() < crossover_rate and len(next_pop) >= 2:
                # Tournament selection
                i1, i2 = np.random.choice(len(next_pop), size=2, replace=False)
                p1, p2 = next_pop[i1].copy(), next_pop[i2].copy()
                # Uniform crossover
                mask = np.random.random(n_interventions) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = next_pop[np.random.randint(len(next_pop))].copy()

            # Mutation: flip bits
            for j in range(n_interventions):
                if np.random.random() < mutation_rate:
                    child[j] = 1 - child[j]

            # Enforce package size constraint
            if child.sum() > max_package_size:
                active = np.where(child == 1)[0]
                remove = np.random.choice(active, size=int(child.sum() - max_package_size), replace=False)
                child[remove] = 0

            offspring.append(child)

        population = next_pop + offspring

    # Final evaluation and Pareto front extraction
    objectives = [_evaluate_package(ind, interventions, country_state, prereqs)
                  for ind in population]
    fronts = _fast_non_dominated_sort(objectives)

    # Extract first front (Pareto-optimal)
    pareto_solutions = []
    for idx in fronts[0]:
        package = [interventions[i]["tech_id"] for i, s in enumerate(population[idx]) if s]
        package_names = [interventions[i]["name"] for i, s in enumerate(population[idx]) if s]
        pareto_solutions.append({
            "package": package,
            "package_names": package_names,
            "n_interventions": len(package),
            "objectives": {
                "sdg_score": round(objectives[idx][0], 4),
                "co2_reduction": round(objectives[idx][1], 4),
                "feasibility": round(objectives[idx][2], 4),
                "cost_efficiency": round(objectives[idx][3], 4),
            },
        })

    # Sort by CO₂ reduction (primary) then SDG (secondary)
    pareto_solutions.sort(key=lambda x: (-x["objectives"]["co2_reduction"],
                                         -x["objectives"]["sdg_score"]))

    # Deduplicate by package content
    seen = set()
    unique = []
    for sol in pareto_solutions:
        key = tuple(sorted(sol["package"]))
        if key not in seen:
            seen.add(key)
            unique.append(sol)

    return unique


def run_nsga2_recommendations(featured_df, interventions, output_dir,
                              case_study_countries=("POL", "DEU", "IND")):
    """
    Run NSGA-II optimization for each case study country.
    Saves Pareto front and comparison with weighted-sum solution.
    """
    print("[NSGA-II] Running multi-objective optimization...")

    all_pareto = {}
    for iso3 in case_study_countries:
        country_data = featured_df[featured_df["country_iso3"] == iso3].sort_values("year")
        if country_data.empty:
            continue
        latest = country_data.iloc[-1].to_dict()

        pareto = nsga2_optimize(interventions, latest,
                                pop_size=80, n_generations=40, max_package_size=8)

        all_pareto[iso3] = {
            "n_pareto_solutions": len(pareto),
            "pareto_front": pareto[:15],  # top 15 solutions
        }
        print(f"  {iso3}: {len(pareto)} Pareto-optimal packages found")

    with open(f"{output_dir}/nsga2_pareto.json", "w") as f:
        json.dump(all_pareto, f, indent=2, default=str)

    print(f"[NSGA-II] Results saved to {output_dir}/nsga2_pareto.json")
    return all_pareto
