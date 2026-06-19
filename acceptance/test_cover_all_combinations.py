import pytest

from sweetpea import *
from sweetpea._internal.constraint import CoverAllCombinations


def _stroop(n):
    """Build a Stroop-style inner block with n colors/words and a derived congruency
    factor, crossing [congruency, color] (word left free)."""
    colors = ['red', 'green', 'blue', 'yellow', 'purple'][:n]
    color = Factor('color', colors)
    word = Factor('word', colors)
    congruency = Factor('congruency', [
        DerivedLevel('con', WithinTrial(lambda c, w: c == w, [color, word])),
        DerivedLevel('incon', WithinTrial(lambda c, w: c != w, [color, word])),
    ])
    inner = CrossBlock([congruency, color, word], [congruency, color], [])
    return colors, color, word, congruency, inner


def _stroop_nest(n, name=None):
    colors, color, word, congruency, inner = _stroop(n)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    nest = Nest(outer, inner, [CoverAllCombinations(color, word, name=name)])
    return colors, color, word, nest


# ~~~~~~~~~~~~ K computation (isolated) ~~~~~~~~~~~~

@pytest.mark.parametrize('n,expected_k', [(3, 2), (4, 3), (5, 4)])
def test_required_instances_stroop(n, expected_k):
    _, color, word, _, inner = _stroop(n)
    assert CoverAllCombinations(color, word).required_instances(inner) == expected_k


def test_required_instances_overlap_below_biggest_group():
    # Two free factors, neither crossed: slots fully overlap, so K = ceil(4/2) = 2,
    # not the naive "biggest group" count of 4.
    colr = Factor('colr', ['red', 'green'])
    size = Factor('size', ['big', 'small'])
    task = Factor('task', ['A', 'B'])
    inner = CrossBlock([task, colr, size], [task], [])
    assert CoverAllCombinations(colr, size).required_instances(inner) == 2


# ~~~~~~~~~~~~ Auto-sizing + coverage (end to end) ~~~~~~~~~~~~

@pytest.mark.parametrize('n,expected_trials', [(3, 12), (4, 32)])
def test_autosize_trial_count(n, expected_trials):
    _, _, _, nest = _stroop_nest(n)
    assert nest.trials_per_sample() == expected_trials


@pytest.mark.parametrize('n', [3, 4])
def test_coverage_holds(n):
    colors, _, _, nest = _stroop_nest(n)
    all_pairs = set((c, w) for c in colors for w in colors)
    exps = synthesize_trials(nest, 5, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert set(zip(e['color'], e['word'])) == all_pairs


# ~~~~~~~~~~~~ Validation / error paths ~~~~~~~~~~~~

def test_error_outer_factor_in_scope():
    _, color, word, _, inner = _stroop(3)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    with pytest.raises(ValueError):
        Nest(outer, inner, [CoverAllCombinations(instance, color)])


def test_error_overlap_sequential():
    _, color, word, _, inner = _stroop(3)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    with pytest.raises(ValueError):
        Nest(outer, inner, [Sequential(word), CoverAllCombinations(color, word)])


def test_error_overlap_latin_square():
    _, color, word, _, inner = _stroop(3)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    with pytest.raises(ValueError):
        Nest(outer, inner, [LatinSquare([color, word], name='P'),
                            CoverAllCombinations(color, word)])


def test_error_not_in_nest():
    _, color, word, congruency, _ = _stroop(3)
    with pytest.raises(ValueError):
        CrossBlock([congruency, color, word], [congruency, color],
                   [CoverAllCombinations(color, word)])


def test_error_weighted_levels():
    colors = ['red', 'green', 'blue']
    color = Factor('color', colors)
    word = Factor('word', [Level('red', 2), Level('green'), Level('blue')])
    congruency = Factor('congruency', [
        DerivedLevel('con', WithinTrial(lambda c, w: c == w, [color, word])),
        DerivedLevel('incon', WithinTrial(lambda c, w: c != w, [color, word])),
    ])
    inner = CrossBlock([congruency, color, word], [congruency, color], [])
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    with pytest.raises(ValueError):
        Nest(outer, inner, [CoverAllCombinations(color, word)])
