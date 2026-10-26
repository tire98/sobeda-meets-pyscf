# sobeda-meets-pyscf

This repository holds a script of a [PySCF](https://pyscf.org) implementation of the **sobEDA** and **sobEDAw**
energy decomposition analyses (Lu, T.; Chen, Q. *J. Phys. Chem. A* **2023**, 127 (33), 7023–7035. [doi:10.1021/acs.jpca.3c04374](https://doi.org/10.1021/acs.jpca.3c04374)). The [original scripts and tutorials](http://sobereva.com/soft/sobEDA_tutorial.zip) rely on Gaussian and Multiwfn and are thus limited to the functionals and basis set available in Gaussian. `sobeda.py` is a single script alternative to `sobEDA.sh` with more flexibility in the choice of functional. The key benefits include:

- No Gaussian licence needed.
- Any PySCF/libxc functional except double hybrids, including range-separated hybrids and VV10 functionals such as ωB97M-V.
- Optional GPU acceleration through [gpu4pyscf](https://github.com/pyscf/gpu4pyscf).
- For more than two fragments, an additional pairwise and many-body breakdown.

It reproduces the `sobEDA.sh` tutorial results to the printed precision
(see [Validation](#validation)).

## Method in brief

The methodology is thoroughly introduced in *J. Phys. Chem. A* **2023**, 127 (33), 7023–7035. [doi:10.1021/acs.jpca.3c04374](https://doi.org/10.1021/acs.jpca.3c04374). In short, sobEDA is an energy decomposition analysis scheme that breaks down the interaction energy between arbitrarily defined fragments of an interaction complex into different physically motivated terms that allow an intuitive interpretation of key interaction terms: electrostatics, exchange, Pauli repulsion, orbital relaxation, correlation and dispersion.

In contrast to other energy decomposition schemes, sobEDA computes these terms using dispersion corrected density functional approximations and may be more affordable for large molecules than, e.g., perturbation theory approaches (SAPT).

The individual interaction terms are computed in four steps:
```
isolated fragments A, B, ...    SCF of each fragment (in the complex basis, see below)
      │  P = Σ_A P_A            sum of fragment densities, no antisymmetrisation
      ▼
promolecule                     ΔE_els, ΔE_x, ΔE_DFTc, ΔE_dc  (changes of each component)
      │  Löwdin-orthonormalise the occupied fragment orbitals
      ▼
frozen state                    ΔE_rep = E(frozen) − E(promolecule)
      │  SCF of the whole system (frozen state as initial guess)
      ▼
final state                     ΔE_orb = E(final) − E(frozen)
```

The interaction energy is decomposed into up to six physically meaningful components:

$$\Delta E_\mathrm{int} = \Delta E_\mathrm{els} + \Delta E_\mathrm{x} + \Delta E_\mathrm{rep} + \Delta E_\mathrm{orb} + \Delta E_\mathrm{DFTc} + \Delta E_\mathrm{dc}$$

- $\Delta E_\mathrm{els}$ – electrostatic energy: the classical Coulomb interaction between the unperturbed, undistorted fragment densities of the promolecule.
- $\Delta E_\mathrm{x}$ – exchange energy: the quantum-mechanical exchange energy arising from the indistinguishability of electrons on antisymmetrisation of the fragment wavefunctions, as defined by the exchange part of the exchange-correlation functional.
- $\Delta E_\mathrm{rep}$ – Pauli repulsion: the steric, overlap-driven repulsion introduced by Löwdin orthogonalisation of the occupied fragment orbitals in forming the frozen state.
- $\Delta E_\mathrm{orb}$ – orbital interaction energy: the stabilisation gained on allowing the frozen-state orbitals to relax self-consistently to the fully converged supermolecule, implicitly including polarisation and charge-transfer effects.
- $\Delta E_\mathrm{DFTc}$ – DFT correlation energy: the correlation contribution to the interaction energy, as defined by the correlation part of the exchange-correlation functional.
- $\Delta E_\mathrm{dc}$ – dispersion correction: the contribution from the dispersion-correction model (e.g. D3(BJ)), where separable from the exchange-correlation energy (see below).

The kinetic energy does not change between the fragments and the
promolecule, so the first step contains only the four component terms.

**sobEDAw** splits `ΔE_DFTc` between dispersion and exchange-repulsion,
giving SAPT-like terms for weak interactions (eqs 3–5 of the paper):

$$w = \min\left[1,\ (1-c)\,e^{-r(\Delta E_\mathrm{dc}/\Delta E_\mathrm{els} - a)} + c\right]$$

$$\Delta E_\mathrm{disp} = \Delta E_\mathrm{dc} + w\,\Delta E_\mathrm{DFTc}, \qquad \Delta E_\mathrm{xrep}^\mathrm{w} = \Delta E_\mathrm{x} + \Delta E_\mathrm{rep} + (1-w)\,\Delta E_\mathrm{DFTc}$$

The parameters `c, a, r` need to fitted for each functional and basis set so that `ΔE_disp/ΔE_els` reproduces DFT-SAPT for the S66 set (Tables 1–2 of the paper; listed [below](#sobedaw-parameters)). So far, fitted parameters exist only for B3LYP, BLYP, BHandHLYP, TPSS and TPSSh with D3(BJ); `sobeda.py` warns for any other functional. sobEDAw is not meant for interactions with strong covalent character, where it exaggerates dispersion ([see original paper](https://doi.org/10.1021/acs.jpca.3c04374)).

### Complex basis (counterpoise)

Every fragment SCF uses the *complex basis* (CB): the basis functions of the full complex, using ghost atoms in place of the other fragments' basis functions. This is the Boys–Bernardi counterpoise correction and corresponds to `iCP=1` in `sobEDA.sh`. `sobeda.py` has no monomer-basis (MB, `iCP=0`) mode. Using the CB is more expensive and may lead to convergence issues for individual fragments, but is the more accurate approach especially for weak interactions and an incomplete basis set. With CB, `ΔE_int` equals the counterpoise-corrected interaction energy.

- **Size.** For H₂O···NH₃ at B3LYP/6-311+G(2d,p) the BSSE is 0.55 kcal/mol, about 8 % of `ΔE_int`. It is larger for smaller basis sets, especially those without diffuse functions.
- **sobEDAw.** Only the CB parameters ("w.CB" in Tables 1–2) apply here.
- **Cost.** Each fragment SCF is as large as the SCF of the whole complex.
- **Scope.** Only the basis is corrected. Fragments are computed at their geometry in the complex, so no deformation energy is included (as in `sobEDA.sh`).

### Exluded Terms

This energy decomposition scheme and this script do not compute deformation, solvation, and thermal corrections terms. As mentioned in the paper and `sobEDA.sh` these need to be computed separately with ordinary single point calculations and added if needed.

## Differences from `sobEDA.sh`

| | `sobEDA.sh` | `sobeda.py` |
|---|---|---|
| Programs | Gaussian 16 + Multiwfn | PySCF, pyscf-dispersion, optionally gpu4pyscf |
| Fragment basis | monomer basis (`iCP=0`) or complex basis (`iCP=1`) | complex basis only (≡ `iCP=1`, see [Complex basis](#complex-basis-counterpoise)) |
| Spin | open-shell fragments, spin flip | closed-shell (RKS) only; fragments may be charged |
| Functionals | those for which Gaussian prints energy components; no range-separated | any except double hybrids; six-term sobEDA for the functionals in `XC_SPLIT` (BLYP, B3LYP, BHandHLYP, PBE, PBE0, TPSS, TPSSh, M06-L, M06, M06-2X, MN15), five-term otherwise (below) |
| Basis sets, ECPs | Gaussian's, incl. `genecp` | PySCF's library (and [basis-set-exchange](https://github.com/MolSSI/basis_set_exchange) if installed); per-element basis sets and ECPs (below) |
| Dispersion | D3(BJ) (`em=gd3bj`) | D3(BJ), D3(0), D4; switched off for VV10 functionals |
| Extras | `idetail` stage breakdown | pairwise + many-body terms, density fitting, GPU, Molden/JSON output, optional density cubes |

### Five-term variant (extension, not in the paper)

libxc defines some functionals, e.g. ωB97M-V and B97M-V, as a single exchange-correlation functional, so `ΔE_x` and `ΔE_DFTc` cannot be separated in libxc/PySCF (the B97-type form is separable in principle). For any functional not listed in `XC_SPLIT` in `sobeda.py`, the script therefore reports the combined `ΔE_xc`. For information it also prints the semilocal, VV10 nonlocal and exact-exchange parts of `ΔE_xc`, and `ΔE_xcrep = ΔE_xc + ΔE_rep`.

These numbers are **not** directly comparable to sobEDA's `ΔE_x`/`ΔE_xrep`, and sobEDAw is undefined for them.

> **Caution:** the semilocal, VV10 and exact-exchange parts are printed for information only, and none of them has a clear physical meaning on its own. The functional's parameters were fitted for the sum, so the parts compensate each other. In particular, the VV10 part is **not** the dispersion energy: the semilocal part of such functionals also carries mid-range correlation, and VV10 also contributes outside the dispersion regime. Interpret `ΔE_xc` and `ΔE_xcrep`, not their components. The paper notes that VV10 is in principle compatible with sobEDA as the dispersion term (p. 7024); here it stays inside `ΔE_xc` because it is an integral part of these functionals. VV10 is evaluated on a coarser grid (`NLC_GRID_LEVEL = 2`; PySCF's default is 3), the same for all stages.

### Pairwise and many-body terms (extension, not in the paper)

With more than two fragments, every pair ($i$, $j$) gets one extra SCF in the full-cluster basis, with all other fragments as ghost atoms. Its promolecule and frozen state are built from the fragment results already computed, and the many-body part of each term is `total − Σ pairs`. Because the pairs use the full-cluster basis, the pair terms differ slightly from a stand-alone dimer calculation. `--no-pairwise` skips this step.

## Installation

Python ≥ 3.9 (tested with 3.9 and PySCF 2.14.0). Create an isolated
environment, for example with Conda:

```bash
conda create -n sobeda python=3.11 pip
conda activate sobeda
```

Conda provides the Python interpreter and environment; pip installs this
project and the dependencies declared in `pyproject.toml`. Conda is optional:
a standard Python `venv` works too if you already have a suitable interpreter.

With the environment activated, install the project:

```bash
python -m pip install git+https://github.com/tire98/sobeda-meets-pyscf
# or from a clone:
python -m pip install .
```

`python -m pip` uses the active environment's interpreter. Install any
Conda packages before using pip, and avoid switching between the two to
manage the same dependencies. If you later need additional Conda packages,
recreate the environment with those packages first, then repeat the pip
installation ([Conda guidance](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment)).

This installs PySCF, pyscf-dispersion and the `sobeda` command. Optional
extras: `.[gpu]` for gpu4pyscf (CUDA 12; for CUDA 11 install
`gpu4pyscf-cuda11x` instead), `.[bse]` for basis sets and ECPs not in
PySCF's library, `.[test]` for the tests.

`sobeda.py` also runs without installation, as
`python sobeda.py ...`, once PySCF and pyscf-dispersion are installed.

## Usage

The input files follow `sobEDA.sh`: `system.xyz` (geometry, Å) and
`fragment.txt`. The level of theory, which `sobEDA.sh` takes from
`template.gjf`, is given on the command line:

```bash
sobeda system.xyz fragment.txt --xc B3LYP --basis def2-TZVP \
    --disp d3bj --sobedaw 0.538 0.066 2.724 > output.out
```

The results and messages go to stdout, so redirect them to a file as above (add `2>&1` to include warnings).

`fragment.txt` has the number of fragments on line 1, then two lines per
fragment: `charge multiplicity`, and the atom indices (1-based,
Multiwfn-style ranges separated by commas or spaces):

```
2
0 1
1-3,7,10-12
1 1
4-6,8,9
```

| Option | Meaning |
|---|---|
| `--xc` | functional (PySCF/libxc name; required; see the [PySCF functional documentation](https://pyscf.org/user/dft.html#predefined-xc-functionals-and-functional-aliases) and the [libxc list](https://libxc.gitlab.io/functionals/)) |
| `--basis SPEC …` | basis set `NAME` and/or per-element `El=NAME` (required; see the [PySCF basis set documentation](https://pyscf.org/user/gto.html#basis-set)), e.g. `def2-TZVP` or `'6-311G**' Cu=stuttgartrsc` |
| `--ecp SPEC …` | ECP `NAME` and/or `El=NAME` (built-in ECPs: [PySCF table](https://pyscf.org/user/gto.html#ecp)); default: the ECP that belongs to each element's basis set, if any (see below) |
| `--disp {d3bj,d3zero,d4}` | dispersion correction (default: none) |
| `--sobedaw C A R` | also print sobEDAw terms with these parameters |
| `--cart` | Cartesian d, f, … functions. Gaussian uses Cartesian d (6D) but spherical f (7F) for the 6-31G family, so this matches Gaussian only for 6-31G-type bases without f functions, e.g. 6-31+G(d,p) |
| `--grid-level N` | PySCF DFT grid level. Default 7: 90 radial × 974 angular points for H, 135 × 1202 for Li–Ne; Gaussian's UltraFine is a pruned 99 × 590 |
| `--df {off,rijk,rij}` | density fitting |
| `--no-pairwise` | skip the pairwise / many-body breakdown (>2 fragments) |
| `--gpu`, `--threads N` | gpu4pyscf backend; OpenMP threads |
| `--outdir DIR` | output directory (default `output/`) |
| `--no-chk`, `--no-molden` | do not keep the `.chk` checkpoint files (no restart) / do not write the `.molden` files |
| `--cubes` | also write a density cube file for every stage |
| `--verbose N`, `--logfile FILE` | PySCF verbosity (default 4); write progress to `FILE` and each SCF's log to `DIR/*.pyscf_out` |

Set the memory limit through PySCF's environment variable, e.g.
`export PYSCF_MAX_MEMORY=16000` (MB).

The script also works as a library: `sobeda.sobeda(atoms, fragments, xc, basis, ...)` returns the terms in Hartree, and `sobeda.sobedaw(terms, c, a, r)` applies sobEDAw. All inputs are validated before the first SCF.

### Basis sets and ECPs

By default every element gets the ECP that belongs to its basis set, if PySCF has one: def2 sets from Rb onwards, LANL2DZ, the Stuttgart sets (`stuttgartrsc`, `stuttgartdz`, …) and CRENBL. Gaussian and ORCA do the same for def2 and LANL2DZ basis keywords. Other elements stay all-electron. `--ecp` overrides this, e.g. `--ecp Cu=stuttgartrsc`. Ghost atoms never carry an ECP. Gaussian's SDD is not in PySCF's library; PySCF's Stuttgart sets or basis-set-exchange provide Stuttgart–Dresden ECPs, but check that they match the variant you want.

### Choosing a level

Available functionals: the [PySCF documentation](https://pyscf.org/user/dft.html#predefined-xc-functionals-and-functional-aliases) explains how functionals are named and combined, and the [libxc list](https://libxc.gitlab.io/functionals/) gives all of them (double hybrids are not supported by `sobeda.py`). Available basis sets: see the [PySCF documentation](https://pyscf.org/user/gto.html#basis-set); most built-in sets come from the [Basis Set Exchange](https://www.basissetexchange.org/), and basis-set-exchange (optional) gives access to the rest of its library.

There is no generally best model chemistry; the most suitable functional and basis set depend on the system. Base the choice on functional benchmarks that compare computed properties with experiment or with a systematically improvable method such as coupled cluster, and follow established practice for basis sets (see also M. Bursch, J. Mewes, A. Hansen and S. Grimme, Angew. Chem. Int. Ed., 2022, 61, e202205735.). Always weigh accuracy against cost, and compare results obtained using the same model chemistry. For sobEDAw, the choice is limited to the functionals with fitted parameters ([sobEDAw parameters](#sobedaw-parameters)).

### Output

`DIR/` contains a `.chk` and a `.molden` file for each fragment (`frag1`, `frag2`, …) and the final state, a `.chk` file for each fragment pair (`pair1-2`, …), and `sobeda_summary.json` with all terms (kcal/mol), the stage energies and the consistency checks. `--cubes` adds density cube files, including the promolecule and the frozen state; `--logfile` adds one `.pyscf_out` log per SCF. Fragments are numbered from 1, in the order of `fragment.txt`.

An existing `.chk` file is used as the initial guess only if it belongs to the same molecule, charge and basis; otherwise it is ignored and overwritten. `--no-chk` and `--no-molden` skip the respective files; without `.chk` files a run cannot be restarted.

### Consistency checks

Every run prints:

- the largest difference between each SCF energy and the same energy rebuilt from its components. This is the meaningful check: it fails if the exchange/correlation split does not match the SCF functional.
- `ΔT_1` (kinetic energy change from fragments to promolecule) and the closure error (sum of the terms minus `ΔE_int`). Both are zero by construction and only guard against bookkeeping errors.

### sobEDAw parameters

Complex-basis parameters `c, a, r` from Tables 1 and 2 of the paper, all with D3(BJ). The first five match the values given in `sobEDA.sh`.

| Functional | 6-31+G(d,p) | 6-311+G(2d,p) | def2-TZVP |
|---|---|---|---|
| B3LYP | 0.575, 0.071, 2.571 | 0.550, 0.037, 2.750 | 0.538, 0.066, 2.724 |
| BHandHLYP | 0.634, −0.065, 0.702 | 0.595, −0.284, 0.582 | 0.680, −0.050, 1.216 |
| BLYP | 0.338, −0.033, 2.783 | 0.310, −0.047, 2.860 | 0.304, −0.026, 2.800 |
| TPSS | 0.675, 0.556, 1.125 | 0.677, 0.422, 1.345 | 0.698, 0.488, 1.429 |
| TPSSh | 0.519, 0.693, 0.730 | 0.531, 0.567, 0.806 | 0.583, 0.629, 0.934 |

More B3LYP-D3(BJ) complex-basis sets from Table 1: 6-31G(d,p) 0.424, 0.152, 0.829; 6-311+G(d,p) 0.553, 0.058, 2.633; def2-TZVP(−f) 0.546, 0.064, 2.594; ma-def2-TZVP(−f) 0.538, 0.050, 2.690. (−f) means without the f functions of main-group elements. The 6-31G-family values were fitted with Gaussian, which uses Cartesian d functions for these basis sets, so use `--cart` with them. Parameters for other levels can be fitted with the authors' script [sobEDAw_fit.zip](http://sobereva.com/soft/sobEDAw_fit.zip) (paper, p. 7029).

## Examples

| Example | System | Level | Notes |
|---|---|---|---|
| [`examples/HOH-NH3`](examples/HOH-NH3) | H₂O···NH₃, 2 fragments | B3LYP-D3(BJ)/6-311+G(2d,p) | `run.sh`; `run_wb97mv.sh` repeats it with ωB97M-V (five-term, no reference) |
| [`examples/water_tetramer`](examples/water_tetramer) | (H₂O)₄, 4 fragments | BHandHLYP-D3(BJ)/6-31+G(d,p) | `run.sh`, including the pairwise / many-body breakdown |

`system.xyz`, `fragment.txt`, `template.gjf` and `reference_result.txt` (the original `result.txt`) are unchanged copies from the `sobEDA.sh` tutorial (see [Credits](#credits)). `template.gjf` is not read by `sobeda.py`; it is kept to document the reference level of theory.

## Validation

`sobeda.py` (PySCF 2.14.0, CPU, grid level 7) compared with the `sobEDA.sh` reference outputs. All values in kcal/mol; the reference is printed to 0.01.

**H₂O···NH₃**, B3LYP-D3(BJ)/6-311+G(2d,p):

| Term | `sobEDA.sh` | `sobeda.py` |
|---|---:|---:|
| ΔE_int | −7.18 | −7.182 |
| ΔE_els | −12.15 | −12.146 |
| ΔE_x | −6.40 | −6.403 |
| ΔE_rep | 20.28 | 20.284 |
| ΔE_orb | −5.38 | −5.383 |
| ΔE_DFTc | −2.70 | −2.702 |
| ΔE_dc | −0.83 | −0.833 |
| sobEDAw w | 96.26 % | 96.26 % |
| sobEDAw ΔE_disp | −3.43 | −3.434 |

**(H₂O)₄**, BHandHLYP-D3(BJ)/6-31+G(d,p) (`--cart`):

| Term | `sobEDA.sh` | `sobeda.py` |
|---|---:|---:|
| ΔE_int | −33.15 | −33.147 |
| ΔE_els | −57.56 | −57.562 |
| ΔE_x | −46.49 | −46.492 |
| ΔE_rep | 116.98 | 116.976 |
| ΔE_orb | −28.90 | −28.897 |
| ΔE_DFTc | −14.62 | −14.615 |
| ΔE_dc | −2.56 | −2.555 |
| sobEDAw w | 97.29 % | 97.29 % |
| sobEDAw ΔE_disp | −16.78 | −16.775 |

From a clone, install the test dependencies and run the suite in the active environment:

```bash
python -m pip install '.[test]'
python -m pytest
```

It checks every term to ±0.05 kcal/mol, together with fast unit tests on small systems, and takes about 5 min on 4 CPU cores.

The [GitHub Actions workflow](.github/workflows/tests.yml) runs the full CPU
test suite on Ubuntu with Python 3.11 for pushes and pull requests. It also
checks the installed `sobeda` command. You can start a run manually from the
repository's **Actions → Tests → Run workflow** menu once the workflow is on
the default branch.

## Citing

If you use this code, cite the sobEDA paper:

- T. Lu, Q. Chen, *J. Phys. Chem. A* **2023**, 127, 7023.
  [doi:10.1021/acs.jpca.3c04374](https://doi.org/10.1021/acs.jpca.3c04374)

Also cite the methods and software it builds on, and the functional and basis set you chose. Refer to the suggested references in the PySCF generated output file.

- PySCF: Q. Sun, *et al.*,  *J. Chem. Phys.*, **2020**, 153, 024109.
- libxc: S. Lehtola, C. Steigemann, M. Oliveira and M. Marques, *SoftwareX*, **2018**, 7, 1–5.
- gpu4pyscf (`--gpu`): Li, R., Sun, Q., Zhang, X. & Chan, G. K.-L. Introducing GPU-acceleration into the Python-based Simulations of Chemistry Framework. arXiv (2024) doi:10.48550/arxiv.2407.09700.
- Counterpoise correction: S. F. Boys and F. Bernardi, *Mol. Phys.*, **1970**, 19, 553–566.
- D3(BJ): S. Grimme, J. Antony, S. Ehrlich and H. Krieg,  *J. Chem. Phys.*, **2010**, 132, 154104.
- D4: E. Caldeweyher, C. Bannwarth and S. Grimme,  *J. Chem. Phys.*, **2017**, 147, 034112; E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth and S. Grimme,  *J. Chem. Phys.*, **2019**, 150, 154122.
- VV10: O. A. Vydrov and T. V. Voorhis,  *J. Chem. Phys.*, **2010**, 133, 244103.

sobEDA was developed in the Multiwfn ecosystem. Citing Multiwfn (T. Lu,  *J. Chem. Phys.*, **2024**, 161, 082503.) acknowledges the origin of the method and of the examples.

## Credits

The sobEDA/sobEDAw method and [`sobEDA.sh`](http://sobereva.com/soft/sobEDA_tutorial.zip) are by Tian Lu. The example inputs and reference outputs in `examples/` come unchanged from his *Tutorial of using sobEDA.sh to perform sobEDA and sobEDAw analyses* ([sobEDA_tutorial.zip](http://sobereva.com/soft/sobEDA_tutorial.zip)). This repository contains no code from `sobEDA.sh` or Multiwfn.
