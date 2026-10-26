"""Tests for sobeda.py.

Regression tests against the sobEDA.sh (Multiwfn + Gaussian 16) results
shipped with the examples (examples/*/reference_result.txt), plus fast unit
tests on small systems. Run with: pytest tests/
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, NoReturn

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sobeda  # noqa: E402

TOL = 0.05    # kcal/mol; the reference is printed to 0.01 kcal/mol

HE3 = [('He', (0.0, 0.0, 3.0 * k)) for k in range(3)]
FAST: dict[str, Any] = dict(verbose=0, grid_level=3)


def frags(*atom_lists: list[int],
          charges: list[int] | None = None) -> list[dict]:
    """Build singlet fragments from 0-based atom lists; default to neutral."""
    charges = charges or [0] * len(atom_lists)
    return [{'atoms': list(a), 'charge': q, 'mult': 1}
            for a, q in zip(atom_lists, charges)]


# ---------------------------------------------------------------- input files

def test_read_fragments(tmp_path: pathlib.Path) -> None:
    """Parse the fragment.txt example of section 2.4 of the tutorial."""
    p = tmp_path / 'fragment.txt'
    p.write_text('2\n0 1\n1-3,7,10-12\n1 2\n4-6,8,9\n')
    assert sobeda.read_fragments(str(p)) == [
        {'atoms': [0, 1, 2, 6, 9, 10, 11], 'charge': 0, 'mult': 1},
        {'atoms': [3, 4, 5, 7, 8], 'charge': 1, 'mult': 2},
    ]


@pytest.mark.parametrize('text', ['', '2\n0 1\n1\n', '1\n0 1\n1\n0 1\n2\n'],
                         ids=['empty', 'too-few-lines', 'too-many-lines'])
def test_read_fragments_errors(tmp_path: pathlib.Path, text: str) -> None:
    """Reject empty files and a fragment count that does not match."""
    p = tmp_path / 'fragment.txt'
    p.write_text(text)
    with pytest.raises(ValueError):
        sobeda.read_fragments(str(p))


def test_parse_indices() -> None:
    """Accept commas and whitespace as separators; reject bad tokens."""
    assert sobeda._parse_indices('1-3 7') == [0, 1, 2, 6]
    assert sobeda._parse_indices(' 1, 2 ,4-5 ') == [0, 1, 3, 4]
    for bad in ('a', '0', '3-1', '1--2', '1-'):
        with pytest.raises(ValueError):
            sobeda._parse_indices(bad)


def test_parse_spec() -> None:
    """Convert CLI basis/ECP tokens into a name or per-element dict."""
    assert sobeda._parse_spec(['def2-SVP']) == 'def2-SVP'
    assert sobeda._parse_spec(['6-311G*', 'CU=stuttgartrsc']) == {
        'default': '6-311G*', 'Cu': 'stuttgartrsc'}
    with pytest.raises(ValueError):
        sobeda._parse_spec(['a', 'b'])


# ------------------------------------------------- validation before any SCF

@pytest.mark.parametrize('frag_atoms, message', [
    ([[0], [1]], r'missing \[3\]'),
    ([[0, 1], [1, 2]], r'in several fragments \[2\]'),
    ([[0], [1, 2, 3]], r'out of range \[4\]'),
], ids=['missing', 'duplicate', 'out-of-range'])
def test_rejects_invalid_partition(monkeypatch: pytest.MonkeyPatch,
                                   frag_atoms: list[list[int]],
                                   message: str) -> None:
    """Reject invalid partitions before SCF; errors use 1-based indices."""
    monkeypatch.setattr(sobeda, '_make_ks', _no_scf)
    with pytest.raises(ValueError, match=message):
        sobeda.sobeda(HE3, frags(*frag_atoms), 'HF', 'sto-3g', verbose=0)


def _no_scf(*args: Any, **kwargs: Any) -> NoReturn:
    """Stand in for _make_ks and fail if any SCF is set up."""
    raise AssertionError('SCF started before input validation finished')


@pytest.mark.parametrize('atoms, fragments, kwargs', [
    (HE3, frags([0, 1, 2]), {}),
    (HE3, [{'atoms': [0]}, {'atoms': [1, 2], 'charge': 0}], {}),
    (HE3, [{'atoms': [0], 'charge': 0, 'mult': 3},
           {'atoms': [1, 2], 'charge': 0}], {}),
    ([('He', (0, 0, 0)), ('H', (0, 0, 3))], frags([0], [1]), {}),
    (HE3, frags([0], [1, 2]), {'disp': 'd3foo'}),
    (HE3, frags([0], [1, 2]), {'use_df': 'rij'}),
    (HE3, frags([0], [1, 2]), {'xc': 'WB97X', 'sobedaw_parm': (1, 0, 1)}),
    ([('He', (0, 0, 0)), ('Xe', (0, 0, 4))], frags([0], [1]),
     {'basis': 'def2-svp', 'ecp': {'Xe': 'sto-3g'}}),
], ids=['one-fragment', 'missing-charge', 'triplet', 'odd-electrons',
        'bad-disp', 'bad-use_df', 'sobedaw-five-term', 'unknown-ecp'])
def test_rejects_before_scf(monkeypatch: pytest.MonkeyPatch,
                            atoms: list[tuple], fragments: list[dict],
                            kwargs: dict) -> None:
    """Reject invalid inputs before any SCF is set up."""
    monkeypatch.setattr(sobeda, '_make_ks', _no_scf)
    args: dict[str, Any] = dict(xc='B3LYP', basis='sto-3g', verbose=0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        sobeda.sobeda(atoms, fragments, **args)


# ------------------------------------------------------- small calculations

@pytest.mark.parametrize('xc', sorted(sobeda.XC_SPLIT) + ['WB97X', 'WB97M-V'])
def test_reconstruction(xc: str) -> None:
    """Rebuild SCF energies for separable, range-separated and VV10 XC."""
    terms, checks, _ = sobeda.sobeda(HE3[:2], frags([0], [1]), xc, '6-31g',
                                     **FAST)
    assert checks['recon_max_abs'] < 1e-8
    assert abs(checks['sum_minus_int']) < 1e-10
    assert ('E_x' in terms) == (xc in sobeda.XC_SPLIT)
    assert ('E_xc' in terms) == (xc not in sobeda.XC_SPLIT)


def test_vv10_drops_dispersion() -> None:
    """Drop D3 for VV10 functionals, so E_dc is exactly zero."""
    terms, _, _ = sobeda.sobeda(HE3[:2], frags([0], [1]), 'WB97M-V', '6-31g',
                                disp='d3bj', **FAST)
    assert terms['E_dc'] == 0.0
    assert terms['E_xc_nlc'] != 0.0
    with pytest.raises(ValueError):
        sobeda.sobedaw(terms, 0.5, 0.0, 1.0)


def test_pairwise() -> None:
    """Check the pairwise terms close and add up with the many-body part."""
    terms, _, pairs = sobeda.sobeda(HE3, frags([0], [1], [2]), 'B3LYP',
                                    '6-31g', disp='d3bj', **FAST)
    assert pairs is not None
    assert sorted(pairs['pairs']) == [(0, 1), (0, 2), (1, 2)]
    for t in pairs['pairs'].values():
        assert abs(sobeda._sum_check(t)) < 1e-10
    for k, v in terms.items():
        total = sum(t[k] for t in pairs['pairs'].values())
        assert total + pairs['many_body'][k] == pytest.approx(v, abs=1e-12)


def test_ecp_and_per_element_basis() -> None:
    """Use the def2 ECP of Xe automatically and include it in E_els."""
    atoms = [('He', (0.0, 0.0, 0.0)), ('Xe', (0.0, 0.0, 4.0))]
    basis = {'default': '6-31g', 'Xe': 'def2-svp'}
    terms, checks, _ = sobeda.sobeda(atoms, frags([0], [1]), 'B3LYP', basis,
                                     **FAST)
    assert checks['frag_Etot'][1] > -1000    # all-electron Xe: ~ -7230 Ha
    assert checks['recon_max_abs'] < 1e-8


def test_outputs_and_restart(tmp_path: pathlib.Path) -> None:
    """Write outputs to a path with '%', restart from them, ignore stale chk."""
    outdir = str(tmp_path / 'out%1')
    run: dict[str, Any] = dict(save_outputs=True, save_cubes=True,
                               outdir=outdir, **FAST)
    first, _, _ = sobeda.sobeda(HE3[:2], frags([0], [1]), 'B3LYP', '6-31g',
                                **run)
    with open(os.path.join(outdir, 'sobeda_summary.json')) as fh:
        summary = json.load(fh)
    assert summary['terms_kcal_per_mol']['E_int'] == pytest.approx(
        first['E_int'] * sobeda.H2KCAL)
    assert os.path.exists(os.path.join(outdir, 'frozen_density.cube'))

    again, _, _ = sobeda.sobeda(HE3[:2], frags([0], [1]), 'B3LYP', '6-31g',
                                **run)
    assert again['E_int'] == pytest.approx(first['E_int'], abs=1e-9)

    moved = [HE3[0], ('He', (0.0, 0.0, 2.5))]
    stale, _, _ = sobeda.sobeda(moved, frags([0], [1]), 'B3LYP', '6-31g',
                                **run)
    fresh, _, _ = sobeda.sobeda(moved, frags([0], [1]), 'B3LYP', '6-31g',
                                **FAST)
    assert stale['E_int'] == pytest.approx(fresh['E_int'], abs=1e-9)


def test_cli(tmp_path: pathlib.Path,
             capsys: pytest.CaptureFixture[str]) -> None:
    """Run the command line with a VV10 functional and a D3 request."""
    (tmp_path / 'system.xyz').write_text(
        '2\nHe2\nHe 0 0 0\nHe 0 0 3\n')
    (tmp_path / 'fragment.txt').write_text('2\n0 1\n1\n0 1\n2\n')
    outdir = tmp_path / 'out'
    sobeda.main([str(tmp_path / 'system.xyz'), str(tmp_path / 'fragment.txt'),
                 '--xc', 'WB97M-V', '--basis', '6-31g', '--disp', 'd3bj',
                 '--grid-level', '3', '--verbose', '0',
                 '--outdir', str(outdir)])
    out = capsys.readouterr().out
    assert 'D3BJ' not in out and 'not applied' in out
    assert 'ΔE_int' in out
    assert (outdir / 'sobeda_summary.json').exists()
    assert not list(outdir.glob('*.cube'))


# ------------------------------------------- regression against sobEDA.sh

# Values from"Final results" sections of the original sobEDA.sh outputs
# below (paths relative to the repository root).
# Energies are in kcal/mol; weights are fractions, not percentages.
CASES = [
    # examples/HOH-NH3/reference_result.txt
    pytest.param(
        'HOH-NH3',
        dict(xc='B3LYP', basis='6-311+G(2d,p)', disp='d3bj'),
        (0.550, 0.037, 2.750),
        {'E_int': -7.18, 'E_els': -12.15, 'E_x': -6.40,
         'E_rep': 20.28, 'E_orb': -5.38, 'E_DFTc': -2.70, 'E_dc': -0.83},
        {'E_els': -12.15, 'E_xrep': 13.78, 'E_orb': -5.38, 'E_disp': -3.43},
        0.9626,
        id='HOH-NH3',
    ),
    # examples/water_tetramer/reference_result.txt
    pytest.param(
        'water_tetramer',
        dict(xc='BHANDHLYP', basis='6-31+G(d,p)', disp='d3bj', cart=True),
        (0.634, -0.065, 0.702),
        {'E_int': -33.15, 'E_els': -57.56, 'E_x': -46.49,
         'E_rep': 116.98, 'E_orb': -28.90, 'E_DFTc': -14.62, 'E_dc': -2.56},
        {'E_els': -57.56, 'E_xrep': 70.09, 'E_orb': -28.90, 'E_disp': -16.78},
        0.9729,
        id='water_tetramer',
    ),
]


@pytest.mark.parametrize('name, level, parm, expected, expected_w, weight', CASES)
def test_against_sobeda_sh(name: str, level: dict,
                           parm: tuple[float, float, float],
                           expected: dict, expected_w: dict,
                           weight: float) -> None:
    """Match the tutorial's sobEDA and sobEDAw reference terms."""
    d = os.path.join(ROOT, 'examples', name)
    atoms = sobeda.read_xyz(os.path.join(d, 'system.xyz'))
    fragments = sobeda.read_fragments(os.path.join(d, 'fragment.txt'))
    terms, checks, _ = sobeda.sobeda(atoms, fragments, verbose=0,
                                     pairwise=False, **level)

    assert abs(checks['dT1']) < 1e-8
    assert abs(checks['sum_minus_int']) < 1e-8
    assert checks['recon_max_abs'] < 1e-6

    for key, value in expected.items():
        assert terms[key] * sobeda.H2KCAL == pytest.approx(
            value, abs=TOL), key

    tw = sobeda.sobedaw(terms, *parm)
    assert tw['w'] == pytest.approx(weight, abs=1e-3)
    for key, value in expected_w.items():
        assert tw[key] * sobeda.H2KCAL == pytest.approx(
            value, abs=TOL), key
