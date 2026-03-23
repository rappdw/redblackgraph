"""
RedBlackGraph family DAG synthesizer.

Generates large, realistic family-structure DAGs with controllable properties
for testing and benchmarking.

Pairing model:
- Same-generation pairings (both parents from gen N-1) are matched first
  and are the most common.
- Cross-generation pairings (gen N-1 with gen N-2) happen as a second
  phase, with a proclivity for older males pairing with younger females.
  Max gap between partners is 1 generation.
- 2-generation gaps between partners never occur.

Monogamy model (divorce/remarriage):
- pct_monogamous controls the fraction of pairings that are lifelong (both
  partners are removed from the eligible pool permanently).
- Non-monogamous individuals pair at most once per generation cycle (one
  partner at a time) but remain eligible in subsequent cycles, modeling
  divorce/remarriage.  Over their 2-cycle eligibility window they may
  pair with up to two different partners in successive generations.
- Eligibility window: individuals from generation G are eligible to pair in
  generation cycles G+1 and G+2 (a ~2-generation reproductive window).
  After that they age out of the eligible pool.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix

from redblackgraph.constants import RED_ONE, BLACK_ONE
from redblackgraph.sparse import rb_matrix
from .names import NameGenerator


@dataclass
class SynthesizerConfig:
    """Configuration for family DAG synthesis."""
    num_initial_nodes: int
    pct_red: float
    avg_children_per_pairing: float
    num_generations: int
    pct_monogamous: float
    pct_non_procreating: float
    seed: Optional[int] = None
    max_total_vertices: Optional[int] = None
    consanguinity_depth: int = 3
    pct_immigration_per_gen: float = 0.0
    child_distribution: str = 'poisson'
    child_dispersion: float = 3.0

    def __post_init__(self):
        if self.num_initial_nodes < 1:
            raise ValueError(f"num_initial_nodes must be >= 1, got {self.num_initial_nodes}")
        if not 0 <= self.pct_red <= 100:
            raise ValueError(f"pct_red must be in [0, 100], got {self.pct_red}")
        if self.avg_children_per_pairing < 0:
            raise ValueError(f"avg_children_per_pairing must be >= 0, got {self.avg_children_per_pairing}")
        if self.num_generations < 0:
            raise ValueError(f"num_generations must be >= 0, got {self.num_generations}")
        if not 0 <= self.pct_monogamous <= 100:
            raise ValueError(f"pct_monogamous must be in [0, 100], got {self.pct_monogamous}")
        if not 0 <= self.pct_non_procreating <= 100:
            raise ValueError(f"pct_non_procreating must be in [0, 100], got {self.pct_non_procreating}")
        if self.max_total_vertices is not None and self.max_total_vertices < 1:
            raise ValueError(f"max_total_vertices must be >= 1, got {self.max_total_vertices}")
        if self.consanguinity_depth not in (2, 3):
            raise ValueError(f"consanguinity_depth must be 2 or 3, got {self.consanguinity_depth}")
        if self.pct_immigration_per_gen < 0:
            raise ValueError(f"pct_immigration_per_gen must be >= 0, got {self.pct_immigration_per_gen}")
        if self.child_distribution not in ('poisson', 'negative_binomial'):
            raise ValueError(f"child_distribution must be 'poisson' or 'negative_binomial', got {self.child_distribution!r}")
        if self.child_dispersion <= 0:
            raise ValueError(f"child_dispersion must be > 0, got {self.child_dispersion}")


@dataclass
class Individual:
    """A person in the synthesized family DAG."""
    vertex_id: int
    color: int
    generation: int
    first_name: str
    last_name: str
    father_id: Optional[int] = None
    mother_id: Optional[int] = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class SynthesisResult:
    """Result of family DAG synthesis."""
    matrix: rb_matrix
    individuals: list
    stats: dict


class FamilyDagSynthesizer:
    """Synthesize realistic family-structure DAGs."""

    def __init__(self, config: SynthesizerConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.name_gen = NameGenerator(self.rng)
        self.individuals: list[Individual] = []
        self.edges: list[tuple[int, int]] = []  # (child_id, parent_id)
        self.parent_ids: dict[int, set[int]] = {}
        self.grandparent_ids: dict[int, set[int]] = {}
        self.great_grandparent_ids: dict[int, set[int]] = {}
        self.generation_members: dict[int, list[int]] = {}
        self.monogamous_committed: set[int] = set()
        self.non_procreating: set[int] = set()
        # Per-generation pairing statistics
        self._gen_stats: dict[int, dict] = {}
        # Aggregate pairing statistics
        self._total_pairings = 0
        self._total_childless_pairings = 0
        self._total_children_from_pairings = 0
        self._consanguinity_rejections = 0
        self._consanguinity_checks = 0

    def synthesize(self) -> SynthesisResult:
        """Run the full synthesis and return results."""
        self._create_generation_zero()
        for gen in range(1, self.config.num_generations + 1):
            if self._at_population_cap():
                break
            self._synthesize_generation(gen)
            self._add_immigrants(gen)
        matrix = self._build_matrix()
        self._validate_graph(matrix)
        return SynthesisResult(
            matrix=matrix,
            individuals=list(self.individuals),
            stats=self._compute_stats(),
        )

    def _at_population_cap(self) -> bool:
        cap = self.config.max_total_vertices
        return cap is not None and len(self.individuals) >= cap

    def _create_generation_zero(self):
        n = self.config.num_initial_nodes
        n_red = round(n * self.config.pct_red / 100)

        colors = [RED_ONE] * n_red + [BLACK_ONE] * (n - n_red)
        self.rng.shuffle(colors)

        for i, color in enumerate(colors):
            if color == RED_ONE:
                first, last = self.name_gen.random_male_name()
            else:
                first, last = self.name_gen.random_female_name()

            ind = Individual(
                vertex_id=i, color=color, generation=0,
                first_name=first, last_name=last,
            )
            self.individuals.append(ind)
            self.parent_ids[i] = set()
            self.grandparent_ids[i] = set()
            self.great_grandparent_ids[i] = set()

        self.generation_members[0] = list(range(n))

        # Mark non-procreating
        n_non_proc = round(n * self.config.pct_non_procreating / 100)
        if n_non_proc > 0:
            indices = self.rng.choice(n, size=n_non_proc, replace=False)
            self.non_procreating.update(indices)

    def _synthesize_generation(self, gen_num: int):
        """Synthesize one generation of pairings and offspring.

        Pairing priority (models real demographics):
        1. Same-generation pairings (both from gen N-1) — most common
        2. Cross-generation pairings (gen N-1 with gen N-2) — less common,
           with a proclivity for older males / younger females

        Each individual pairs at most once per generation cycle.  Monogamous
        individuals are permanently removed from the eligible pool.
        Non-monogamous individuals pair once this cycle but remain eligible
        in subsequent cycles (remarriage).  Max gap between partners is
        1 generation.
        """
        new_children_ids = []
        used_males = set()
        used_females = set()
        gen_consanguinity_checks = 0
        gen_consanguinity_rejections = 0
        gen_same_gen_pairings = 0
        gen_cross_gen_pairings = 0
        gen_pairings = 0
        gen_childless = 0

        # --- Phase 1: same-generation pairings (gen N-1 with gen N-1) ---
        same_gen = gen_num - 1
        if same_gen >= 0:
            males_sg, females_sg = self._eligible_from_gen(same_gen)
            self.rng.shuffle(males_sg)
            self.rng.shuffle(females_sg)
            stats = self._pair_lists(
                males_sg, females_sg, gen_num,
                new_children_ids, used_males, used_females,
            )
            gen_consanguinity_checks += stats['checks']
            gen_consanguinity_rejections += stats['rejections']
            gen_same_gen_pairings += stats['pairings']
            gen_pairings += stats['pairings']
            gen_childless += stats['childless']

        # --- Phase 2: cross-generation pairings (gen N-2 with gen N-1) ---
        # Only gen N-2 individuals not yet paired; they pair with remaining
        # gen N-1 individuals.  Max gap is 1 generation.
        older_gen = gen_num - 2
        if older_gen >= 0:
            males_og, females_og = self._eligible_from_gen(older_gen)
            # Remaining gen N-1 partners
            remaining_males_sg = [m for m in males_sg if m not in used_males]
            remaining_females_sg = [f for f in females_sg if f not in used_females]

            # Older males with younger females (more common pattern)
            self.rng.shuffle(males_og)
            self.rng.shuffle(remaining_females_sg)
            stats = self._pair_lists(
                males_og, remaining_females_sg, gen_num,
                new_children_ids, used_males, used_females,
            )
            gen_consanguinity_checks += stats['checks']
            gen_consanguinity_rejections += stats['rejections']
            gen_cross_gen_pairings += stats['pairings']
            gen_pairings += stats['pairings']
            gen_childless += stats['childless']

            # Older females with younger males (less common but possible)
            remaining_males_sg = [m for m in males_sg if m not in used_males]
            self.rng.shuffle(females_og)
            self.rng.shuffle(remaining_males_sg)
            stats = self._pair_lists(
                remaining_males_sg, females_og, gen_num,
                new_children_ids, used_males, used_females,
            )
            gen_consanguinity_checks += stats['checks']
            gen_consanguinity_rejections += stats['rejections']
            gen_cross_gen_pairings += stats['pairings']
            gen_pairings += stats['pairings']
            gen_childless += stats['childless']

        self.generation_members[gen_num] = new_children_ids

        # Store per-generation stats
        self._gen_stats[gen_num] = {
            'pairings': gen_pairings,
            'same_gen_pairings': gen_same_gen_pairings,
            'cross_gen_pairings': gen_cross_gen_pairings,
            'childless_pairings': gen_childless,
            'consanguinity_checks': gen_consanguinity_checks,
            'consanguinity_rejections': gen_consanguinity_rejections,
            'consanguinity_rejection_pct': (
                gen_consanguinity_rejections / gen_consanguinity_checks * 100
                if gen_consanguinity_checks > 0 else 0.0
            ),
        }

    def _add_immigrants(self, gen_num: int):
        """Add unrelated immigrants to the current generation."""
        if self.config.pct_immigration_per_gen <= 0:
            return
        n_immigrants = round(
            self.config.num_initial_nodes * self.config.pct_immigration_per_gen / 100
        )
        if n_immigrants <= 0:
            return

        # Respect population cap
        cap = self.config.max_total_vertices
        if cap is not None:
            remaining = cap - len(self.individuals)
            n_immigrants = min(n_immigrants, max(0, remaining))
        if n_immigrants <= 0:
            return

        n_red = round(n_immigrants * self.config.pct_red / 100)
        colors = [RED_ONE] * n_red + [BLACK_ONE] * (n_immigrants - n_red)
        self.rng.shuffle(colors)

        for color in colors:
            vid = len(self.individuals)
            if color == RED_ONE:
                first, last = self.name_gen.random_male_name()
            else:
                first, last = self.name_gen.random_female_name()

            ind = Individual(
                vertex_id=vid, color=color, generation=gen_num,
                first_name=first, last_name=last,
            )
            self.individuals.append(ind)
            self.parent_ids[vid] = set()
            self.grandparent_ids[vid] = set()
            self.great_grandparent_ids[vid] = set()

            # Append to current generation so they're eligible next cycle
            self.generation_members[gen_num].append(vid)

            # Mark non-procreating at same rate
            if self.rng.random() < self.config.pct_non_procreating / 100:
                self.non_procreating.add(vid)

    def _eligible_from_gen(self, gen: int) -> tuple:
        """Return (males, females) eligible from a single generation."""
        males = []
        females = []
        for vid in self.generation_members.get(gen, []):
            if vid in self.non_procreating:
                continue
            if vid in self.monogamous_committed:
                continue
            ind = self.individuals[vid]
            if ind.color == RED_ONE:
                males.append(vid)
            else:
                females.append(vid)
        return males, females

    def _pair_lists(
        self, males, females, gen_num,
        new_children_ids, used_males, used_females,
    ) -> dict:
        """Greedily pair males with females, one pairing per individual.

        Returns dict with pairing statistics for this batch.
        """
        checks = 0
        rejections = 0
        pairings = 0
        childless = 0

        for male in males:
            if self._at_population_cap():
                break
            if male in used_males:
                continue
            for female in females:
                if female in used_females:
                    continue
                checks += 1
                self._consanguinity_checks += 1
                if not self._check_consanguinity(male, female):
                    rejections += 1
                    self._consanguinity_rejections += 1
                    continue
                # Form pairing — one per individual per cycle
                used_males.add(male)
                used_females.add(female)
                is_mono = self.rng.random() < self.config.pct_monogamous / 100
                if is_mono:
                    self.monogamous_committed.add(male)
                    self.monogamous_committed.add(female)
                children = self._create_pairing(male, female, gen_num)
                pairings += 1
                if len(children) == 0:
                    childless += 1
                new_children_ids.extend(children)
                break

        return {
            'checks': checks,
            'rejections': rejections,
            'pairings': pairings,
            'childless': childless,
        }

    def _check_consanguinity(self, male_id: int, female_id: int) -> bool:
        """Return True if pairing is allowed (no shared recent ancestors).

        Checks up to consanguinity_depth generations:
        - depth 2: rejects siblings and first cousins (shared parents/grandparents)
        - depth 3: also rejects second cousins (shared great-grandparents)
        """
        if male_id == female_id:
            return False
        # Parent-child
        if male_id in self.parent_ids.get(female_id, set()):
            return False
        if female_id in self.parent_ids.get(male_id, set()):
            return False
        # Siblings (share any parent)
        m_parents = self.parent_ids.get(male_id, set())
        f_parents = self.parent_ids.get(female_id, set())
        if m_parents and f_parents and m_parents & f_parents:
            return False
        # First cousins (share any grandparent)
        m_gp = self.grandparent_ids.get(male_id, set())
        f_gp = self.grandparent_ids.get(female_id, set())
        if m_gp and f_gp and m_gp & f_gp:
            return False
        # Second cousins (share any great-grandparent) — depth >= 3
        if self.config.consanguinity_depth >= 3:
            m_ggp = self.great_grandparent_ids.get(male_id, set())
            f_ggp = self.great_grandparent_ids.get(female_id, set())
            if m_ggp and f_ggp and m_ggp & f_ggp:
                return False
        return True

    def _sample_child_count(self) -> int:
        """Sample number of children for a pairing from configured distribution."""
        mean = self.config.avg_children_per_pairing
        if self.config.child_distribution == 'negative_binomial':
            n = self.config.child_dispersion
            p = n / (n + mean)
            return int(self.rng.negative_binomial(n, p))
        return int(self.rng.poisson(mean))

    def _create_pairing(self, male_id: int, female_id: int, gen_num: int) -> list:
        """Generate children for a pairing. Returns list of child vertex IDs."""
        n_children = self._sample_child_count()

        # Respect population cap
        cap = self.config.max_total_vertices
        if cap is not None:
            remaining = cap - len(self.individuals)
            n_children = min(n_children, max(0, remaining))

        self._total_pairings += 1
        if n_children == 0:
            self._total_childless_pairings += 1
            return []

        father_surname = self.individuals[male_id].last_name
        children = []

        for _ in range(n_children):
            child_id = len(self.individuals)
            child_color = self.rng.choice([RED_ONE, BLACK_ONE])

            if child_color == RED_ONE:
                first, last = self.name_gen.random_male_name(surname=father_surname)
            else:
                first, last = self.name_gen.random_female_name(surname=father_surname)

            child = Individual(
                vertex_id=child_id, color=child_color, generation=gen_num,
                first_name=first, last_name=last,
                father_id=male_id, mother_id=female_id,
            )
            self.individuals.append(child)
            children.append(child_id)

            # Edges: child -> father, child -> mother
            self.edges.append((child_id, male_id))
            self.edges.append((child_id, female_id))

            # Track ancestry
            self.parent_ids[child_id] = {male_id, female_id}
            self.grandparent_ids[child_id] = (
                self.parent_ids.get(male_id, set()) |
                self.parent_ids.get(female_id, set())
            )
            self.great_grandparent_ids[child_id] = (
                self.grandparent_ids.get(male_id, set()) |
                self.grandparent_ids.get(female_id, set())
            )

            # Mark non-procreating
            if self.rng.random() < self.config.pct_non_procreating / 100:
                self.non_procreating.add(child_id)

        self._total_children_from_pairings += len(children)
        return children

    def _build_matrix(self) -> rb_matrix:
        """Build sparse CSR matrix from individuals and edges (vectorized)."""
        n = len(self.individuals)
        m = len(self.edges)

        # Diagonal entries: vertex colors
        diag_indices = np.arange(n, dtype=np.int32)
        diag_data = np.array([ind.color for ind in self.individuals], dtype=np.int32)

        if m > 0:
            # Edge entries: child -> parent with value 2 (male) or 3 (female)
            edge_arr = np.array(self.edges, dtype=np.int32)  # shape (m, 2)
            edge_rows = edge_arr[:, 0]
            edge_cols = edge_arr[:, 1]
            parent_colors = np.array(
                [self.individuals[pid].color for pid in edge_cols], dtype=np.int32
            )
            edge_data = np.where(parent_colors == RED_ONE, 2, 3).astype(np.int32)

            data = np.concatenate([diag_data, edge_data])
            rows = np.concatenate([diag_indices, edge_rows])
            cols = np.concatenate([diag_indices, edge_cols])
        else:
            data = diag_data
            rows = diag_indices
            cols = diag_indices

        coo = coo_matrix((data, (rows, cols)), shape=(n, n))
        return rb_matrix(coo)

    def _validate_graph(self, matrix: rb_matrix):
        """Validate the synthesized graph satisfies AVOS structural invariants.

        Uses sparse iteration — O(nnz), not O(n²) — so it scales to large
        graphs without dense memory allocation.

        Raises ValueError if any invariant is violated.
        """
        n = matrix.shape[0]

        # Validate diagonal entries via sparse diagonal
        diag = matrix.diagonal()
        for i in range(n):
            if diag[i] not in (RED_ONE, BLACK_ONE):
                raise ValueError(
                    f"Vertex {i}: diagonal is {diag[i]}, "
                    f"expected {RED_ONE} or {BLACK_ONE}"
                )

        # Validate edges: correct values and DAG ordering
        for child_id, parent_id in self.edges:
            if child_id <= parent_id:
                raise ValueError(
                    f"Edge ({child_id},{parent_id}): child_id must be > parent_id"
                )
            parent_color = diag[parent_id]
            expected = 2 if parent_color == RED_ONE else 3
            actual = matrix[child_id, parent_id]
            if actual != expected:
                raise ValueError(
                    f"Edge ({child_id},{parent_id}): value {actual}, "
                    f"expected {expected} for parent color {parent_color}"
                )

        # Validate all non-zero off-diagonal entries are 2 or 3
        # Uses sparse COO iteration — O(nnz) not O(n²)
        coo = matrix.tocoo()
        for r, c, v in zip(coo.row, coo.col, coo.data):
            if r == c:
                continue
            if v not in (2, 3):
                raise ValueError(f"M[{r},{c}] = {v}, expected 2 or 3")

        # Expected nnz: n diagonal + len(edges) off-diagonal
        expected_nnz = n + len(self.edges)
        if coo.nnz != expected_nnz:
            raise ValueError(
                f"Matrix has {coo.nnz} non-zeros, "
                f"expected {expected_nnz} ({n} diagonal + {len(self.edges)} edges)"
            )

        # Every child has exactly 2 parents or 0 parents
        for ind in self.individuals:
            has_father = ind.father_id is not None
            has_mother = ind.mother_id is not None
            if has_father != has_mother:
                raise ValueError(
                    f"Vertex {ind.vertex_id}: has "
                    f"{'father' if has_father else 'mother'} but not "
                    f"{'mother' if has_father else 'father'}"
                )

    def _compute_half_siblings(self) -> dict:
        """Count half-sibling relationships in the graph."""
        from collections import defaultdict
        children_of_father = defaultdict(set)
        children_of_mother = defaultdict(set)

        for ind in self.individuals:
            if ind.father_id is not None:
                children_of_father[ind.father_id].add(ind.vertex_id)
                children_of_mother[ind.mother_id].add(ind.vertex_id)

        half_sibling_pairs = set()
        # Half-siblings through father (same father, different mother)
        for father_id, children in children_of_father.items():
            if len(children) < 2:
                continue
            mother_groups = defaultdict(set)
            for cid in children:
                mother_groups[self.individuals[cid].mother_id].add(cid)
            groups = list(mother_groups.values())
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    for c1 in groups[i]:
                        for c2 in groups[j]:
                            half_sibling_pairs.add(
                                (min(c1, c2), max(c1, c2))
                            )

        # Half-siblings through mother (same mother, different father)
        for mother_id, children in children_of_mother.items():
            if len(children) < 2:
                continue
            father_groups = defaultdict(set)
            for cid in children:
                father_groups[self.individuals[cid].father_id].add(cid)
            groups = list(father_groups.values())
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    for c1 in groups[i]:
                        for c2 in groups[j]:
                            half_sibling_pairs.add(
                                (min(c1, c2), max(c1, c2))
                            )

        # Count full siblings for comparison
        full_sibling_pairs = set()
        pair_to_children = defaultdict(set)
        for ind in self.individuals:
            if ind.father_id is not None:
                pair_to_children[(ind.father_id, ind.mother_id)].add(ind.vertex_id)
        for children in pair_to_children.values():
            children_list = sorted(children)
            for i in range(len(children_list)):
                for j in range(i + 1, len(children_list)):
                    full_sibling_pairs.add((children_list[i], children_list[j]))

        return {
            'half_sibling_pairs': len(half_sibling_pairs),
            'full_sibling_pairs': len(full_sibling_pairs),
        }

    def _compute_stats(self) -> dict:
        n = len(self.individuals)
        children_with_parents = self._total_children_from_pairings
        fertile_pairings = self._total_pairings - self._total_childless_pairings

        stats = {
            "total_vertices": n,
            "total_edges": len(self.edges),
            "num_generations": len(self.generation_members),
            "generations": {},
            "total_pairings": self._total_pairings,
            "childless_pairings": self._total_childless_pairings,
            "configured_avg_children": self.config.avg_children_per_pairing,
            "realized_avg_children": (
                children_with_parents / self._total_pairings
                if self._total_pairings > 0 else 0.0
            ),
            "realized_avg_children_fertile": (
                children_with_parents / fertile_pairings
                if fertile_pairings > 0 else 0.0
            ),
            "consanguinity_checks": self._consanguinity_checks,
            "consanguinity_rejections": self._consanguinity_rejections,
            "consanguinity_rejection_pct": (
                self._consanguinity_rejections / self._consanguinity_checks * 100
                if self._consanguinity_checks > 0 else 0.0
            ),
            "per_generation": {
                str(gen): gs for gen, gs in self._gen_stats.items()
            },
        }

        # Sibling stats
        sibling_stats = self._compute_half_siblings()
        stats["half_sibling_pairs"] = sibling_stats["half_sibling_pairs"]
        stats["full_sibling_pairs"] = sibling_stats["full_sibling_pairs"]

        for gen, members in sorted(self.generation_members.items()):
            n_red = sum(1 for vid in members if self.individuals[vid].color == RED_ONE)
            n_black = len(members) - n_red
            n_immigrant = sum(
                1 for vid in members
                if self.individuals[vid].father_id is None
                and self.individuals[vid].generation > 0
            )
            stats["generations"][gen] = {
                "total": len(members), "male": n_red, "female": n_black,
                "immigrants": n_immigrant,
            }
        if n > 1:
            stats["matrix_density_pct"] = (n + len(self.edges)) / (n * n) * 100
        else:
            stats["matrix_density_pct"] = 0.0

        # Surname diversity
        gen0_surnames = set(
            ind.last_name for ind in self.individuals if ind.generation == 0
        )
        last_gen = max(self.generation_members.keys())
        last_gen_surnames = set(
            self.individuals[vid].last_name
            for vid in self.generation_members.get(last_gen, [])
        )
        stats["surname_diversity"] = {
            "gen0": len(gen0_surnames),
            f"gen{last_gen}": len(last_gen_surnames),
        }

        return stats


def _prompt(msg: str, default) -> str:
    """Prompt user with a default value."""
    result = input(f"{msg} [{default}]: ").strip()
    return result if result else str(default)


def interactive_config() -> SynthesizerConfig:
    """Prompt user for synthesis parameters."""
    print()
    print("RedBlackGraph Family DAG Synthesizer")
    print("=" * 40)
    print()
    n = int(_prompt("Number of initial nodes (generation 0)", 50))
    pct_red = float(_prompt("% of initial nodes that are red/male", 50))
    avg_children = float(_prompt("Average children per pairing", 2.5))
    generations = int(_prompt("Number of generations to synthesize", 4))
    pct_mono = float(_prompt("% of monogamous pairings", 70))
    pct_non = float(_prompt("% of individuals who never procreate", 15))
    seed_str = input("Random seed (blank for random) []: ").strip()
    seed = int(seed_str) if seed_str else None
    max_str = input("Max total vertices (blank for unlimited) []: ").strip()
    max_verts = int(max_str) if max_str else None
    depth = int(_prompt("Consanguinity depth (2=cousins, 3=second cousins)", 3))
    pct_imm = float(_prompt("% immigration per generation", 0))
    dist = _prompt("Child distribution (poisson/negative_binomial)", "poisson")
    dispersion = 3.0
    if dist == 'negative_binomial':
        dispersion = float(_prompt("Dispersion parameter", 3.0))
    return SynthesizerConfig(
        n, pct_red, avg_children, generations, pct_mono, pct_non, seed,
        max_verts, depth, pct_imm, dist, dispersion,
    )


def print_summary(result: SynthesisResult):
    """Print synthesis summary to stdout."""
    s = result.stats
    print()
    print("Synthesis complete:")
    print(f"  Total vertices:  {s['total_vertices']:>8,}")
    for gen, info in sorted(s["generations"].items()):
        imm = f"  +{info['immigrants']} imm" if info.get('immigrants', 0) > 0 else ""
        print(f"  Generation {gen}:    {info['total']:>8,}"
              f"  ({info['male']} male, {info['female']} female){imm}")
    print(f"  Total edges:     {s['total_edges']:>8,}")
    print(f"  Total pairings:  {s['total_pairings']:>8,}"
          f"  ({s['childless_pairings']} childless)")
    if s['total_pairings'] > 0:
        print(f"  Avg children/pairing: {s['realized_avg_children']:>5.2f}"
              f"  (configured: {s['configured_avg_children']:.1f},"
              f" fertile only: {s['realized_avg_children_fertile']:.2f})")
    if s["consanguinity_checks"] > 0:
        print(f"  Consanguinity checks: {s['consanguinity_checks']:>6,}"
              f"  ({s['consanguinity_rejection_pct']:.1f}% rejected)")
    print(f"  Siblings: {s['full_sibling_pairs']} full, {s['half_sibling_pairs']} half")
    sd = s.get('surname_diversity', {})
    if sd:
        keys = sorted(sd.keys(), key=lambda k: int(k[3:]))
        parts = ", ".join(f"{k}: {sd[k]}" for k in keys)
        print(f"  Surname diversity: {parts}")
    print(f"  Matrix density:  {s['matrix_density_pct']:>7.2f}%")
    print(f"  Matrix shape:    {result.matrix.shape[0]} x {result.matrix.shape[1]}")

    # Per-generation pairing breakdown
    pg = s.get("per_generation", {})
    if pg:
        print()
        print("  Per-generation pairing breakdown:")
        for gen in sorted(pg.keys(), key=lambda k: int(k)):
            gs = pg[gen]
            total_p = gs['pairings']
            if total_p == 0:
                continue
            print(f"    Gen {gen}: {total_p} pairings"
                  f" (same:{gs['same_gen_pairings']}"
                  f" cross:{gs['cross_gen_pairings']}"
                  f" childless:{gs['childless_pairings']})"
                  f"  consang: {gs['consanguinity_rejection_pct']:.1f}% rejected")
    print()


def save_output(result: SynthesisResult, output_path: str):
    """Save synthesis results to files.

    Writes files based on the output_path stem:
    - <stem>.npz — sparse matrix in numpy compressed format
    - <stem>.csv — individual metadata (proper CSV with quoting)
    - <stem>.vertices.csv + <stem>.edges.csv — format compatible with
      RelationshipFileReader for ingestion into the existing pipeline
    - <stem>.stats.json — full statistics as JSON
    """
    stem = output_path.rsplit(".", 1)[0] if "." in output_path else output_path

    # Save sparse matrix
    from scipy.sparse import save_npz
    npz_path = f"{stem}.npz"
    save_npz(npz_path, result.matrix)

    # Save individual metadata (proper CSV with quoting)
    csv_path = f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vertex_id", "first_name", "last_name", "gender",
            "generation", "father_id", "mother_id",
        ])
        for ind in result.individuals:
            gender = "M" if ind.color == RED_ONE else "F"
            father = ind.father_id if ind.father_id is not None else ""
            mother = ind.mother_id if ind.mother_id is not None else ""
            writer.writerow([
                ind.vertex_id, ind.first_name, ind.last_name,
                gender, ind.generation, father, mother,
            ])

    # Save in RelationshipFileReader-compatible format
    # vertices.csv: external_id, color, name, hop
    vert_path = f"{stem}.vertices.csv"
    with open(vert_path, "w", newline="") as f:
        writer = csv.writer(f)
        for ind in result.individuals:
            writer.writerow([
                ind.vertex_id, ind.color, ind.full_name, ind.generation,
            ])

    # edges.csv: source_external_id, destination_external_id, relationship_type
    edge_path = f"{stem}.edges.csv"
    with open(edge_path, "w", newline="") as f:
        writer = csv.writer(f)
        for ind in result.individuals:
            if ind.father_id is not None:
                writer.writerow([ind.vertex_id, ind.father_id, "parent-child"])
            if ind.mother_id is not None:
                writer.writerow([ind.vertex_id, ind.mother_id, "parent-child"])

    # Save stats as JSON
    json_path = f"{stem}.stats.json"
    with open(json_path, "w") as f:
        json.dump(result.stats, f, indent=2)

    print(f"Saved matrix to {npz_path}")
    print(f"Saved metadata to {csv_path}")
    print(f"Saved vertices to {vert_path}")
    print(f"Saved edges to {edge_path}")
    print(f"Saved stats to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize realistic family-structure RedBlackGraph DAGs"
    )
    parser.add_argument("-n", "--initial-nodes", type=int, default=50)
    parser.add_argument("--pct-red", type=float, default=50.0)
    parser.add_argument("--avg-children", type=float, default=2.5)
    parser.add_argument("-g", "--generations", type=int, default=4)
    parser.add_argument("--pct-monogamous", type=float, default=70.0)
    parser.add_argument("--pct-non-procreating", type=float, default=15.0)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--max-vertices", type=int, default=None,
                        help="Maximum total vertices (stops synthesis early)")
    parser.add_argument("--consanguinity-depth", type=int, default=3,
                        choices=[2, 3],
                        help="Ancestor depth for consanguinity check (default: 3)")
    parser.add_argument("--pct-immigration", type=float, default=0.0,
                        help="Percent immigration per generation (based on initial pop)")
    parser.add_argument("--child-distribution", type=str, default='poisson',
                        choices=['poisson', 'negative_binomial'],
                        help="Distribution for child count sampling")
    parser.add_argument("--child-dispersion", type=float, default=3.0,
                        help="Dispersion for negative_binomial (lower = more variance)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (.npz for matrix, .csv for metadata)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Use interactive prompts")

    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        config = interactive_config()
    else:
        config = SynthesizerConfig(
            num_initial_nodes=args.initial_nodes,
            pct_red=args.pct_red,
            avg_children_per_pairing=args.avg_children,
            num_generations=args.generations,
            pct_monogamous=args.pct_monogamous,
            pct_non_procreating=args.pct_non_procreating,
            seed=args.seed,
            max_total_vertices=args.max_vertices,
            consanguinity_depth=args.consanguinity_depth,
            pct_immigration_per_gen=args.pct_immigration,
            child_distribution=args.child_distribution,
            child_dispersion=args.child_dispersion,
        )

    synth = FamilyDagSynthesizer(config)
    result = synth.synthesize()
    print_summary(result)

    if args.output:
        save_output(result, args.output)


if __name__ == "__main__":
    main()
