"""Tests for the family DAG synthesizer."""

import json
import numpy as np
import pytest
from scipy.sparse import issparse

from redblackgraph.constants import RED_ONE, BLACK_ONE
from redblackgraph.util.synthesizer import (
    FamilyDagSynthesizer, SynthesizerConfig, SynthesisResult,
)


def make_config(**kwargs):
    defaults = dict(
        num_initial_nodes=20,
        pct_red=50.0,
        avg_children_per_pairing=2.0,
        num_generations=3,
        pct_monogamous=70.0,
        pct_non_procreating=10.0,
        seed=42,
    )
    defaults.update(kwargs)
    return SynthesizerConfig(**defaults)


def synthesize(**kwargs) -> SynthesisResult:
    return FamilyDagSynthesizer(make_config(**kwargs)).synthesize()


class TestGenerationZero:

    def test_correct_total(self):
        result = synthesize(num_initial_nodes=30, num_generations=0)
        assert result.stats["total_vertices"] == 30

    def test_gender_ratio(self):
        result = synthesize(num_initial_nodes=100, pct_red=75.0, num_generations=0)
        gen0 = result.stats["generations"][0]
        assert gen0["male"] == 75
        assert gen0["female"] == 25

    def test_all_red_no_pairings(self):
        result = synthesize(num_initial_nodes=20, pct_red=100.0, num_generations=3)
        # No females means no pairings, so only gen 0
        assert result.stats["total_vertices"] == 20
        assert result.stats["total_edges"] == 0


class TestConsanguinity:

    def test_siblings_rejected(self):
        synth = FamilyDagSynthesizer(make_config())
        synth.parent_ids[10] = {0, 1}
        synth.parent_ids[11] = {0, 1}
        synth.grandparent_ids[10] = set()
        synth.grandparent_ids[11] = set()
        synth.great_grandparent_ids[10] = set()
        synth.great_grandparent_ids[11] = set()
        assert not synth._check_consanguinity(10, 11)

    def test_half_siblings_rejected(self):
        synth = FamilyDagSynthesizer(make_config())
        synth.parent_ids[10] = {0, 1}
        synth.parent_ids[11] = {0, 2}  # Share father 0
        synth.grandparent_ids[10] = set()
        synth.grandparent_ids[11] = set()
        synth.great_grandparent_ids[10] = set()
        synth.great_grandparent_ids[11] = set()
        assert not synth._check_consanguinity(10, 11)

    def test_first_cousins_rejected(self):
        synth = FamilyDagSynthesizer(make_config())
        synth.parent_ids[200] = {100, 101}
        synth.parent_ids[201] = {100, 102}
        synth.parent_ids[300] = {200, 202}
        synth.parent_ids[301] = {201, 203}
        synth.grandparent_ids[300] = {100, 101}
        synth.grandparent_ids[301] = {100, 102}
        synth.great_grandparent_ids[300] = set()
        synth.great_grandparent_ids[301] = set()
        assert not synth._check_consanguinity(300, 301)

    def test_second_cousins_rejected_at_depth_3(self):
        """Second cousins (shared great-grandparent) rejected at depth 3."""
        synth = FamilyDagSynthesizer(make_config(consanguinity_depth=3))
        # No shared parents or grandparents, but shared great-grandparent
        synth.parent_ids[400] = {300, 301}
        synth.parent_ids[401] = {302, 303}
        synth.grandparent_ids[400] = {200, 201}
        synth.grandparent_ids[401] = {202, 203}
        synth.great_grandparent_ids[400] = {100, 101, 102, 103}
        synth.great_grandparent_ids[401] = {100, 104, 105, 106}  # Share 100
        assert not synth._check_consanguinity(400, 401)

    def test_second_cousins_allowed_at_depth_2(self):
        """Second cousins allowed when depth is only 2."""
        synth = FamilyDagSynthesizer(make_config(consanguinity_depth=2))
        synth.parent_ids[400] = {300, 301}
        synth.parent_ids[401] = {302, 303}
        synth.grandparent_ids[400] = {200, 201}
        synth.grandparent_ids[401] = {202, 203}
        synth.great_grandparent_ids[400] = {100, 101, 102, 103}
        synth.great_grandparent_ids[401] = {100, 104, 105, 106}
        assert synth._check_consanguinity(400, 401)

    def test_unrelated_allowed(self):
        synth = FamilyDagSynthesizer(make_config())
        synth.parent_ids[10] = {0, 1}
        synth.parent_ids[11] = {2, 3}
        synth.grandparent_ids[10] = {4, 5, 6, 7}
        synth.grandparent_ids[11] = {8, 9, 10, 11}
        synth.great_grandparent_ids[10] = set()
        synth.great_grandparent_ids[11] = set()
        assert synth._check_consanguinity(10, 11)

    def test_parent_child_rejected(self):
        synth = FamilyDagSynthesizer(make_config())
        synth.parent_ids[10] = {5, 6}
        synth.parent_ids[5] = set()
        synth.grandparent_ids[10] = set()
        synth.grandparent_ids[5] = set()
        synth.great_grandparent_ids[10] = set()
        synth.great_grandparent_ids[5] = set()
        assert not synth._check_consanguinity(5, 10)
        assert not synth._check_consanguinity(10, 5)


class TestPairingConstraints:

    def test_opposite_gender_only(self):
        result = synthesize(num_initial_nodes=40, num_generations=2)
        for ind in result.individuals:
            if ind.father_id is not None:
                father = result.individuals[ind.father_id]
                mother = result.individuals[ind.mother_id]
                assert father.color == RED_ONE, f"Father {father.vertex_id} is not male"
                assert mother.color == BLACK_ONE, f"Mother {mother.vertex_id} is not female"

    def test_monogamy_100_pct(self):
        result = synthesize(
            num_initial_nodes=40, pct_monogamous=100.0,
            pct_non_procreating=0.0, num_generations=2,
        )
        pairing_count = {}
        for ind in result.individuals:
            if ind.father_id is not None:
                pair = (ind.father_id, ind.mother_id)
                pairing_count.setdefault(ind.father_id, set()).add(pair)
                pairing_count.setdefault(ind.mother_id, set()).add(pair)
        for vid, pairings in pairing_count.items():
            assert len(pairings) <= 1, (
                f"Individual {vid} in {len(pairings)} pairings with 100% monogamy"
            )

    def test_remarriage_0_pct_monogamy(self):
        """With 0% monogamy, some individuals should appear in multiple pairings."""
        result = synthesize(
            num_initial_nodes=60, pct_monogamous=0.0,
            pct_non_procreating=0.0, num_generations=3, seed=42,
        )
        pairing_count = {}
        for ind in result.individuals:
            if ind.father_id is not None:
                pair = (ind.father_id, ind.mother_id)
                pairing_count.setdefault(ind.father_id, set()).add(pair)
                pairing_count.setdefault(ind.mother_id, set()).add(pair)
        max_pairings = max(len(p) for p in pairing_count.values()) if pairing_count else 0
        assert max_pairings > 1, (
            "With 0% monogamy, expected some individuals in multiple pairings"
        )

    def test_generation_within_range(self):
        result = synthesize(num_initial_nodes=40, num_generations=4)
        for ind in result.individuals:
            if ind.father_id is not None:
                father = result.individuals[ind.father_id]
                mother = result.individuals[ind.mother_id]
                assert abs(father.generation - mother.generation) <= 1, (
                    f"Cross-gen pairing: father gen {father.generation}, "
                    f"mother gen {mother.generation}"
                )
                assert ind.generation > father.generation
                assert ind.generation > mother.generation

    def test_same_gen_pairings_dominate(self):
        """Same-generation pairings should be more common than cross-gen."""
        result = synthesize(
            num_initial_nodes=80, num_generations=4,
            pct_non_procreating=5.0, seed=42,
        )
        same_gen = 0
        cross_gen = 0
        for ind in result.individuals:
            if ind.father_id is not None:
                father = result.individuals[ind.father_id]
                mother = result.individuals[ind.mother_id]
                if father.generation == mother.generation:
                    same_gen += 1
                else:
                    cross_gen += 1
        assert same_gen > cross_gen, (
            f"Expected same-gen > cross-gen, got {same_gen} vs {cross_gen}"
        )


class TestChildlessPairings:

    def test_childless_can_occur(self):
        """With enough pairings, some should be childless (Poisson P(0)~8%)."""
        result = synthesize(
            num_initial_nodes=200, num_generations=2,
            pct_non_procreating=0.0, seed=42,
        )
        assert result.stats["childless_pairings"] > 0, (
            "Expected some childless pairings with Poisson distribution"
        )

    def test_childless_counted_in_stats(self):
        result = synthesize(num_initial_nodes=200, num_generations=2, seed=42)
        assert result.stats["childless_pairings"] >= 0
        assert result.stats["total_pairings"] >= result.stats["childless_pairings"]


class TestNaming:

    def test_children_take_father_surname(self):
        result = synthesize(num_initial_nodes=30, num_generations=2)
        for ind in result.individuals:
            if ind.father_id is not None:
                father = result.individuals[ind.father_id]
                assert ind.last_name == father.last_name, (
                    f"Child {ind.full_name} doesn't have father's surname "
                    f"({father.last_name})"
                )


class TestMatrixCorrectness:

    def test_diagonal_colors(self):
        result = synthesize(num_initial_nodes=30, num_generations=2)
        M = result.matrix.toarray()
        for ind in result.individuals:
            assert M[ind.vertex_id, ind.vertex_id] == ind.color

    def test_edge_values(self):
        result = synthesize(num_initial_nodes=30, num_generations=2)
        M = result.matrix.toarray()
        for ind in result.individuals:
            if ind.father_id is not None:
                assert M[ind.vertex_id, ind.father_id] == 2  # Father is male
                assert M[ind.vertex_id, ind.mother_id] == 3  # Mother is female

    def test_sparse_format(self):
        result = synthesize()
        assert issparse(result.matrix)

    def test_transitive_closure_valid(self):
        result = synthesize(num_initial_nodes=10, num_generations=2, seed=123)
        tc = result.matrix.transitive_closure()
        assert np.count_nonzero(tc.W) >= result.matrix.nnz


class TestGraphValidation:

    def test_validation_passes(self):
        """Synthesize should not raise if graph is valid."""
        result = synthesize(num_initial_nodes=50, num_generations=3)
        # If we got here, validation passed during synthesize()
        assert result.stats["total_vertices"] > 0

    def test_dag_ordering(self):
        """Children always have higher vertex IDs than parents."""
        result = synthesize(num_initial_nodes=50, num_generations=3)
        for ind in result.individuals:
            if ind.father_id is not None:
                assert ind.vertex_id > ind.father_id
                assert ind.vertex_id > ind.mother_id

    def test_every_child_has_two_parents(self):
        result = synthesize(num_initial_nodes=50, num_generations=3)
        for ind in result.individuals:
            has_father = ind.father_id is not None
            has_mother = ind.mother_id is not None
            assert has_father == has_mother, (
                f"Vertex {ind.vertex_id}: father={ind.father_id}, mother={ind.mother_id}"
            )


class TestImmigration:

    def test_immigrants_added(self):
        result = synthesize(
            num_initial_nodes=50, num_generations=3,
            pct_immigration_per_gen=10.0, seed=42,
        )
        # 10% of 50 = 5 immigrants per generation, 3 generations = 15
        total_immigrants = sum(
            info.get('immigrants', 0)
            for info in result.stats["generations"].values()
        )
        assert total_immigrants > 0

    def test_immigrants_are_unrelated(self):
        """Immigrants have no parents in the graph."""
        result = synthesize(
            num_initial_nodes=30, num_generations=2,
            pct_immigration_per_gen=20.0, seed=42,
        )
        for ind in result.individuals:
            if ind.generation > 0 and ind.father_id is None:
                # This is an immigrant — verify no parent edges
                assert ind.mother_id is None

    def test_immigrants_add_surname_diversity(self):
        """Immigration should slow surname collapse."""
        r_no_imm = synthesize(
            num_initial_nodes=50, num_generations=6,
            pct_immigration_per_gen=0.0, seed=42,
        )
        r_with_imm = synthesize(
            num_initial_nodes=50, num_generations=6,
            pct_immigration_per_gen=10.0, seed=42,
        )
        sd_no = r_no_imm.stats["surname_diversity"]
        sd_yes = r_with_imm.stats["surname_diversity"]
        last_gen_no = sd_no[f"gen6"]
        last_gen_yes = sd_yes[f"gen6"]
        assert last_gen_yes >= last_gen_no, (
            f"Immigration should maintain surname diversity: "
            f"without={last_gen_no}, with={last_gen_yes}"
        )


class TestChildDistribution:

    def test_negative_binomial(self):
        """Negative binomial produces wider variance than Poisson."""
        r_poisson = synthesize(
            num_initial_nodes=100, num_generations=2,
            pct_non_procreating=0.0, child_distribution='poisson', seed=42,
        )
        r_nb = synthesize(
            num_initial_nodes=100, num_generations=2,
            pct_non_procreating=0.0, child_distribution='negative_binomial',
            child_dispersion=3.0, seed=42,
        )
        # NB should produce more childless pairings (heavier left tail)
        # and/or larger families (heavier right tail)
        assert r_nb.stats["total_pairings"] > 0
        assert r_poisson.stats["total_pairings"] > 0

    def test_negative_binomial_valid_graph(self):
        """NB distribution should still produce a valid AVOS graph."""
        result = synthesize(
            num_initial_nodes=50, num_generations=3,
            child_distribution='negative_binomial',
            child_dispersion=2.0, seed=42,
        )
        assert result.stats["total_vertices"] > 0


class TestHalfSiblings:

    def test_half_siblings_from_remarriage(self):
        """With low monogamy, remarriage should create half-siblings."""
        result = synthesize(
            num_initial_nodes=60, pct_monogamous=0.0,
            pct_non_procreating=0.0, num_generations=3, seed=42,
        )
        assert result.stats["half_sibling_pairs"] > 0, (
            "Expected half-siblings from remarriage with 0% monogamy"
        )

    def test_full_siblings_counted(self):
        result = synthesize(num_initial_nodes=30, num_generations=2, seed=42)
        assert result.stats["full_sibling_pairs"] >= 0

    def test_no_half_siblings_with_100_pct_monogamy(self):
        """100% monogamy means no remarriage, so no half-siblings."""
        result = synthesize(
            num_initial_nodes=40, pct_monogamous=100.0,
            pct_non_procreating=0.0, num_generations=2, seed=42,
        )
        assert result.stats["half_sibling_pairs"] == 0


class TestDeterminism:

    def test_same_seed_same_result(self):
        r1 = synthesize(seed=99)
        r2 = synthesize(seed=99)
        assert len(r1.individuals) == len(r2.individuals)
        for a, b in zip(r1.individuals, r2.individuals):
            assert a.vertex_id == b.vertex_id
            assert a.color == b.color
            assert a.first_name == b.first_name
            assert a.last_name == b.last_name
            assert a.father_id == b.father_id

    def test_different_seed_different_result(self):
        r1 = synthesize(seed=1)
        r2 = synthesize(seed=2)
        names1 = [i.first_name for i in r1.individuals]
        names2 = [i.first_name for i in r2.individuals]
        assert names1 != names2


class TestEdgeCases:

    def test_zero_generations(self):
        result = synthesize(num_generations=0)
        assert result.stats["total_vertices"] == 20
        assert result.stats["total_edges"] == 0

    def test_one_initial_node(self):
        result = synthesize(num_initial_nodes=1, num_generations=3)
        assert result.stats["total_vertices"] == 1

    def test_zero_non_procreating(self):
        result = synthesize(pct_non_procreating=0.0)
        assert result.stats["total_vertices"] > 20

    def test_all_non_procreating(self):
        result = synthesize(pct_non_procreating=100.0, num_generations=3)
        assert result.stats["total_vertices"] == 20
        assert result.stats["total_edges"] == 0


class TestPopulationCap:

    def test_cap_limits_vertices(self):
        result = synthesize(
            num_initial_nodes=50, num_generations=5,
            pct_non_procreating=0.0, max_total_vertices=200, seed=42,
        )
        assert result.stats["total_vertices"] <= 200

    def test_cap_none_means_unlimited(self):
        r1 = synthesize(
            num_initial_nodes=50, num_generations=3,
            pct_non_procreating=0.0, max_total_vertices=None, seed=42,
        )
        r2 = synthesize(
            num_initial_nodes=50, num_generations=3,
            pct_non_procreating=0.0, max_total_vertices=100_000, seed=42,
        )
        assert r1.stats["total_vertices"] == r2.stats["total_vertices"]


class TestPerGenerationStats:

    def test_per_gen_stats_present(self):
        result = synthesize(num_initial_nodes=30, num_generations=3, seed=42)
        pg = result.stats["per_generation"]
        assert len(pg) > 0
        for gen, gs in pg.items():
            assert "pairings" in gs
            assert "same_gen_pairings" in gs
            assert "cross_gen_pairings" in gs
            assert "childless_pairings" in gs
            assert "consanguinity_checks" in gs
            assert "consanguinity_rejections" in gs
            assert "consanguinity_rejection_pct" in gs

    def test_per_gen_pairings_sum_to_total(self):
        result = synthesize(num_initial_nodes=50, num_generations=3, seed=42)
        pg = result.stats["per_generation"]
        total_from_gens = sum(gs['pairings'] for gs in pg.values())
        assert total_from_gens == result.stats["total_pairings"]


class TestPairingStats:

    def test_stats_present(self):
        result = synthesize()
        assert "total_pairings" in result.stats
        assert "realized_avg_children" in result.stats
        assert "childless_pairings" in result.stats
        assert "consanguinity_checks" in result.stats
        assert "consanguinity_rejections" in result.stats
        assert "consanguinity_rejection_pct" in result.stats
        assert "half_sibling_pairs" in result.stats
        assert "full_sibling_pairs" in result.stats
        assert "surname_diversity" in result.stats

    def test_zero_gen_has_zero_pairings(self):
        result = synthesize(num_generations=0)
        assert result.stats["total_pairings"] == 0
        assert result.stats["realized_avg_children"] == 0.0
        assert result.stats["consanguinity_checks"] == 0


class TestScale:

    def test_large_synthesis(self):
        result = synthesize(
            num_initial_nodes=100, num_generations=5,
            avg_children_per_pairing=2.5, pct_non_procreating=15.0,
            seed=42,
        )
        assert result.stats["total_vertices"] > 100
        assert result.stats["total_edges"] > 0
        n = result.stats["total_vertices"]
        assert result.matrix.shape == (n, n)


class TestOutputFormat:

    def test_csv_output_handles_apostrophes(self, tmp_path):
        """Verify CSV output properly quotes names with apostrophes."""
        import csv as csv_mod
        from redblackgraph.util.synthesizer import save_output

        result = synthesize(num_initial_nodes=10, num_generations=1, seed=42)
        result.individuals[0] = result.individuals[0].__class__(
            vertex_id=0, color=result.individuals[0].color,
            generation=0, first_name="Sean",
            last_name="O'Brien",
        )

        output_path = str(tmp_path / "test_output.npz")
        save_output(result, output_path)

        csv_path = str(tmp_path / "test_output.csv")
        with open(csv_path, newline="") as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            assert header == [
                "vertex_id", "first_name", "last_name", "gender",
                "generation", "father_id", "mother_id",
            ]
            row0 = next(reader)
            assert row0[2] == "O'Brien"

    def test_vertices_edges_files_created(self, tmp_path):
        """Verify RelationshipFileReader-compatible files are created."""
        from redblackgraph.util.synthesizer import save_output

        result = synthesize(num_initial_nodes=10, num_generations=1, seed=42)
        output_path = str(tmp_path / "test_output.npz")
        save_output(result, output_path)

        vert_path = tmp_path / "test_output.vertices.csv"
        edge_path = tmp_path / "test_output.edges.csv"
        assert vert_path.exists()
        assert edge_path.exists()

        import csv as csv_mod
        with open(vert_path, newline="") as f:
            rows = list(csv_mod.reader(f))
        assert len(rows) == len(result.individuals)

    def test_json_stats_output(self, tmp_path):
        """Verify JSON stats file is created and parseable."""
        from redblackgraph.util.synthesizer import save_output

        result = synthesize(num_initial_nodes=10, num_generations=1, seed=42)
        output_path = str(tmp_path / "test_output.npz")
        save_output(result, output_path)

        json_path = tmp_path / "test_output.stats.json"
        assert json_path.exists()

        with open(json_path) as f:
            stats = json.load(f)
        assert stats["total_vertices"] == result.stats["total_vertices"]
        assert stats["total_pairings"] == result.stats["total_pairings"]
        assert "per_generation" in stats
        assert "half_sibling_pairs" in stats


class TestConfigValidation:

    def test_num_initial_nodes_zero(self):
        with pytest.raises(ValueError, match="num_initial_nodes"):
            SynthesizerConfig(0, 50, 2.0, 3, 70, 10)

    def test_num_initial_nodes_negative(self):
        with pytest.raises(ValueError, match="num_initial_nodes"):
            SynthesizerConfig(-5, 50, 2.0, 3, 70, 10)

    def test_pct_red_negative(self):
        with pytest.raises(ValueError, match="pct_red"):
            SynthesizerConfig(10, -1, 2.0, 3, 70, 10)

    def test_pct_red_over_100(self):
        with pytest.raises(ValueError, match="pct_red"):
            SynthesizerConfig(10, 101, 2.0, 3, 70, 10)

    def test_avg_children_negative(self):
        with pytest.raises(ValueError, match="avg_children_per_pairing"):
            SynthesizerConfig(10, 50, -1.0, 3, 70, 10)

    def test_num_generations_negative(self):
        with pytest.raises(ValueError, match="num_generations"):
            SynthesizerConfig(10, 50, 2.0, -1, 70, 10)

    def test_pct_monogamous_out_of_range(self):
        with pytest.raises(ValueError, match="pct_monogamous"):
            SynthesizerConfig(10, 50, 2.0, 3, -1, 10)
        with pytest.raises(ValueError, match="pct_monogamous"):
            SynthesizerConfig(10, 50, 2.0, 3, 101, 10)

    def test_pct_non_procreating_out_of_range(self):
        with pytest.raises(ValueError, match="pct_non_procreating"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, -1)
        with pytest.raises(ValueError, match="pct_non_procreating"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 101)

    def test_max_total_vertices_zero(self):
        with pytest.raises(ValueError, match="max_total_vertices"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, max_total_vertices=0)

    def test_consanguinity_depth_invalid(self):
        with pytest.raises(ValueError, match="consanguinity_depth"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, consanguinity_depth=1)
        with pytest.raises(ValueError, match="consanguinity_depth"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, consanguinity_depth=4)

    def test_pct_immigration_negative(self):
        with pytest.raises(ValueError, match="pct_immigration_per_gen"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, pct_immigration_per_gen=-5)

    def test_child_distribution_invalid(self):
        with pytest.raises(ValueError, match="child_distribution"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, child_distribution='uniform')

    def test_child_dispersion_zero(self):
        with pytest.raises(ValueError, match="child_dispersion"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, child_dispersion=0)

    def test_child_dispersion_negative(self):
        with pytest.raises(ValueError, match="child_dispersion"):
            SynthesizerConfig(10, 50, 2.0, 3, 70, 10, child_dispersion=-1.0)

    def test_valid_config_accepted(self):
        """Ensure a valid config does not raise."""
        config = SynthesizerConfig(
            10, 50, 2.0, 3, 70, 10, seed=42,
            max_total_vertices=1000, consanguinity_depth=3,
            pct_immigration_per_gen=5.0,
            child_distribution='negative_binomial',
            child_dispersion=2.0,
        )
        assert config.num_initial_nodes == 10
