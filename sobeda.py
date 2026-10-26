"""sobEDA/sobEDAw energy decomposition analysis with PySCF.

Reimplements the sobEDA scheme of T. Lu and Q. Chen (J. Phys. Chem. A 2023,
127, 7023), originally distributed as sobEDA.sh for Gaussian and Multiwfn.
Fragments are computed in the complex (counterpoise) basis and must be
closed-shell; they may carry net charges.

Functionals in ``XC_SPLIT`` give the six-term decomposition and support
sobEDAw. Any other functional gives a five-term decomposition with a
combined exchange-correlation term. Optional: GPU (gpu4pyscf), density
fitting, ECPs, pairwise/many-body terms for more than two fragments.

Usage::

    python sobeda.py system.xyz fragment.txt --xc B3LYP --basis def2-TZVP
    python sobeda.py --help
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import os
import re
import sys
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from pyscf import dft as _dft_mod  # replaced by gpu4pyscf.dft with --gpu
from pyscf import gto, lib
from pyscf.dft import libxc as _libxc
from pyscf.lib import logger
from pyscf.scf import chkfile as _chkfile
from pyscf.tools import cubegen, molden

if TYPE_CHECKING:
    import cupy
    from pyscf.dft.rks import RKS

H2KCAL = 627.5094740631

# Numerical defaults (the grid level is also a CLI option).
GRID_LEVEL = 7
NLC_GRID_LEVEL = 2   # VV10 scales as O(N_grid^2); PySCF's default is 3
MAX_SCF_CYCLE = 125
CONV_TOL = 1e-8
LINDEP_TOL = 1e-10   # smallest allowed eigenvalue of the occupied-MO overlap
VERBOSE = 4          # PySCF logger level: 3 warn, 4 info, 5+ debug

DISP_VERSIONS = ('d3bj', 'd3zero', 'd4')

# Density fitting. The same auxiliary basis is placed on ghost atoms, so all
# stages share one fitting space (None: PySCF default). The dense 3-index
# tensor is estimated first; above DF_CDERI_LIMIT_GB density fitting is
# switched off instead of running out of memory.
DF_AUXBASIS = 'def2-universal-jkfit'
DF_CDERI_LIMIT_GB = 32

# Exchange/correlation split of separable functionals as libxc strings. Each
# reproduces PySCF's composite functional to machine precision. B3LYP uses
# VWN-RPA (libxc LDA_C_VWN_RPA, PySCF alias 'VWN3'; not libxc's LDA_C_VWN_3),
# as do PySCF's 'B3LYP' and Gaussian.
XC_SPLIT = {
    'BLYP':      ('B88,',                    ',LYP'),
    'B3LYP':     ('0.08*SLATER + 0.72*B88,', ',0.19*VWN3 + 0.81*LYP'),
    'BHANDHLYP': ('0.5*B88,',                ',LYP'),
    'PBE':       ('PBE,',                    ',PBE'),
    'PBE0':      ('0.75*PBE,',               ',PBE'),
    'TPSS':      ('TPSS,',                   ',TPSS'),
    'TPSSH':     ('0.9*TPSS,',               ',TPSS'),
    'M06L':      ('M06L,',                   ',M06L'),
    'M06':       ('M06,',                    ',M06'),
    'M062X':     ('M062X,',                  ',M062X'),
    'MN15':      ('MN15,',                   ',MN15'),
}

# Functionals with fitted sobEDAw parameters (Tables 1-2 of the paper).
# Others, e.g. PBE, PBE0 and M06-2X, partly describe dispersion themselves.
SOBEDAW_FUNCTIONALS = ('B3LYP', 'BLYP', 'BHANDHLYP', 'TPSS', 'TPSSH')


def _canonical_xc(xc: str) -> str:
    """Upper-case a functional name and map spelling variants onto XC_SPLIT.

    Arguments:
        xc {str} -- Functional, e.g. 'b3lyp', 'M06-2X' or 'wB97M-V'.

    Returns:
        {str} -- The XC_SPLIT key if xc matches one ignoring case and
            hyphens ('M06-2X' -> 'M062X'), else xc in upper case.
    """
    xc = xc.upper()
    plain = xc.replace('-', '')
    return plain if plain in XC_SPLIT else xc


def _set_backend(use_gpu: bool) -> None:
    """Select the DFT backend for all subsequent SCF calculations.

    Arguments:
        use_gpu {bool} -- Use gpu4pyscf.dft if usable, else pyscf.dft.
    """
    global _dft_mod
    if use_gpu:
        try:
            from gpu4pyscf import dft as gpu_dft
            _dft_mod = gpu_dft
            return
        except Exception as exc:     # ImportError, or CUDA errors on import
            print(f"Warning: gpu4pyscf unavailable ({exc}); using CPU PySCF.",
                  file=sys.stderr)
    from pyscf import dft as cpu_dft
    _dft_mod = cpu_dft


def read_xyz(path: str) -> list[tuple]:
    """Read the geometry of the whole system from an .xyz file.

    Only the first frame of a multi-frame file is read.

    Arguments:
        path {str} -- Path to an .xyz file (atom count, comment line, one
            "El x y z" line per atom in Angstrom).

    Raises:
        ValueError -- If the file is empty or has fewer coordinate lines
            than atoms.

    Returns:
        {list[tuple]} -- (element, (x, y, z)) for each atom, in Angstrom.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    if not lines or not lines[0].strip():
        raise ValueError(f'{path}: empty file or missing atom count')
    natom = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2:2 + natom]:
        el, x, y, z = ln.split()[:4]
        atoms.append((el, (float(x), float(y), float(z))))
    if len(atoms) != natom:
        raise ValueError(f'{path}: header says {natom} atoms, '
                         f'found {len(atoms)}')
    return atoms


def _parse_indices(spec: str) -> list[int]:
    """Convert a 1-based Multiwfn atom list to 0-based indices.

    Arguments:
        spec {str} -- Indices and ranges separated by commas or spaces,
            e.g. '1-3,7,10-12'.

    Raises:
        ValueError -- If a token is not an index or range, an index is
            below 1, or a range is reversed.

    Returns:
        {list[int]} -- 0-based atom indices in the given order.
    """
    idx: list[int] = []
    for tok in re.split(r'[,\s]+', spec.strip()):
        if not tok:
            continue
        m = re.fullmatch(r'(\d+)(?:-(\d+))?', tok)
        lo = int(m.group(1)) if m else 0
        hi = int(m.group(2)) if m and m.group(2) else lo
        if not m or lo < 1 or hi < lo:
            raise ValueError(f'invalid atom index or range {tok!r}')
        idx.extend(range(lo - 1, hi))
    return idx


def read_fragments(path: str) -> list[dict]:
    """Read a sobEDA.sh fragment definition file.

    Line 1 holds the number of fragments, followed by two lines per
    fragment: "charge multiplicity" and 1-based atom indices in Multiwfn
    syntax (e.g. 1-3,7,10-12). Blank lines are ignored.

    Arguments:
        path {str} -- Path to fragment.txt.

    Raises:
        ValueError -- If the file is empty, or the number of definition
            lines does not match the number of fragments.

    Returns:
        {list[dict]} -- One dict per fragment with keys 'atoms' (list of
            0-based indices), 'charge' (int) and 'mult' (int).
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        raise ValueError(f'{path}: empty file')
    nfrag = int(lines[0].split()[0])
    if len(lines) != 1 + 2 * nfrag:
        raise ValueError(f'{path}: header says {nfrag} fragments, so '
                         f'{2 * nfrag} definition lines are expected, '
                         f'found {len(lines) - 1}')
    fragments = []
    for i in range(nfrag):
        charge, mult = (int(v) for v in lines[1 + 2 * i].split()[:2])
        fragments.append({'atoms': _parse_indices(lines[2 + 2 * i]),
                          'charge': charge, 'mult': mult})
    return fragments


def _element(label: str) -> str:
    """Return the element symbol of an atom label.

    Arguments:
        label {str} -- Atom label such as 'Cu', 'CL' or 'H1'.

    Returns:
        {str} -- Capitalised element symbol, e.g. 'Cu', 'Cl', 'H'.
    """
    m = re.match(r'[A-Za-z]+', label)
    return m.group().capitalize() if m else label


def _name_for(spec: str | dict | None, el: str) -> str | None:
    """Return the basis or ECP name that a specification assigns to an element.

    Arguments:
        spec {str | dict | None} -- One name for all elements, or
            {element: name} with an optional 'default' entry.
        el {str} -- Element symbol.

    Returns:
        {str | None} -- The name, or None if spec assigns none to el.
    """
    if spec is None or isinstance(spec, str):
        return spec
    names = {(k if k == 'default' else _element(k)): v
             for k, v in spec.items()}
    return names.get(el, names.get('default'))


def _spec_str(spec: str | dict | None) -> str:
    """Format a basis or ECP specification for output.

    Arguments:
        spec {str | dict | None} -- Specification as accepted by _name_for.

    Returns:
        {str} -- E.g. 'def2-TZVP' or 'default=6-311G*, Cu=stuttgartrsc'.
    """
    if spec is None or isinstance(spec, str):
        return str(spec)
    return ', '.join(f'{k}={v}' for k, v in spec.items())


def _ecp_data(name: str | None, el: str) -> list | None:
    """Load ECP data from PySCF's library without printing anything.

    Arguments:
        name {str | None} -- ECP or basis set name, e.g. 'def2-svp' or
            'lanl2dz'; None gives None.
        el {str} -- Element symbol.

    Returns:
        {list | None} -- PySCF ECP data [ncore, ...], or None if PySCF has
            no ECP of that name for the element. Any other failure of the
            lookup is reported as a warning before None is returned.
    """
    if not isinstance(name, str):
        return None
    sink = io.StringIO()
    failure = None
    with warnings.catch_warnings(), contextlib.redirect_stdout(sink), \
            contextlib.redirect_stderr(sink):
        warnings.simplefilter('ignore')
        try:
            return gto.basis.load_ecp(name, el) or None
        except KeyError:          # PySCF: no such basis/ECP for the element
            return None
        except Exception as exc:
            failure = exc
    warnings.warn(f'ECP lookup {name!r} for {el} failed ({failure!r}); '
                  f'treating {el} as all-electron.')
    return None


def _ecp_map(atoms: list[tuple], basis: str | dict,
             ecp: str | dict | None) -> dict:
    """Assign ECPs to the elements of the system.

    An element gets the ECP that `ecp` names for it, otherwise the ECP that
    belongs to its basis set if PySCF has one (e.g. def2-* for Z > 36,
    lanl2dz). Elements without ECP data stay all-electron.

    Arguments:
        atoms {list[tuple]} -- (element, (x, y, z)) for each atom.
        basis {str | dict} -- Basis specification (see _name_for).
        ecp {str | dict | None} -- ECP specification (see _name_for);
            None uses the basis sets' own ECPs only.

    Raises:
        ValueError -- If `ecp` names an ECP explicitly for an element and
            PySCF has no such ECP for it.

    Returns:
        {dict} -- {element: ECP name} for every element that gets an ECP.
    """
    explicit = ({_element(k) for k in ecp if k != 'default'}
                if isinstance(ecp, dict) else set())
    out = {}
    for el in sorted({_element(label) for label, _ in atoms}):
        name = _name_for(ecp, el) or _name_for(basis, el)
        if _ecp_data(name, el):
            out[el] = name
        elif el in explicit:
            raise ValueError(f'PySCF has no ECP {name!r} for {el}')
    return out


def _n_electrons(atoms: list[tuple], idx: list[int], charge: int,
                 ecp_map: dict) -> int:
    """Count the electrons of a fragment, excluding ECP core electrons.

    Arguments:
        atoms {list[tuple]} -- (element, (x, y, z)) for each atom.
        idx {list[int]} -- 0-based indices of the fragment's atoms.
        charge {int} -- Net charge of the fragment.
        ecp_map {dict} -- {element: ECP name} as returned by _ecp_map.

    Returns:
        {int} -- Number of explicitly treated electrons.
    """
    nel = -charge
    for i in idx:
        el = _element(atoms[i][0])
        nel += gto.charge(el)
        data = _ecp_data(ecp_map[el], el) if el in ecp_map else None
        nel -= data[0] if data else 0
    return nel


def _to_np(x: Any) -> Any:
    """Return a host copy of a cupy array; pass anything else through.

    Arguments:
        x {Any} -- numpy or cupy array, or a scalar.

    Returns:
        {Any} -- numpy copy if x is a cupy array, else x unchanged.
    """
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return x


def _to_dev(x: np.ndarray) -> np.ndarray | cupy.ndarray:
    """Return a device copy of an array when running on gpu4pyscf.

    Some gpu4pyscf routines (e.g. nr_nlc_vxc) reject numpy input.

    Arguments:
        x {numpy.ndarray} -- Host array.

    Returns:
        {numpy.ndarray | cupy.ndarray} -- cupy copy on gpu4pyscf, else x.
    """
    if _dft_mod.__name__.startswith('gpu4pyscf'):
        import cupy as cp
        return cp.asarray(x)
    return x


def _mol(atoms: list[tuple], ghost_idx: set[int], basis: str | dict,
         charge: int, ecp: dict | None = None, spin: int = 0,
         verbose: int = VERBOSE, output: str | None = None,
         cart: bool = False) -> gto.Mole:
    """Build a Mole in which some atoms are ghosts (basis functions only).

    Arguments:
        atoms {list[tuple]} -- (element, (x, y, z)) for each atom, Angstrom.
        ghost_idx {set[int]} -- 0-based indices of the ghost atoms.
        basis {str | dict} -- Basis set name, or {element: name} with an
            optional 'default' entry.
        charge {int} -- Net charge of the real atoms.

    Keyword Arguments:
        ecp {dict | None} -- {element: ECP name} for the real atoms, as
            returned by _ecp_map; ghost atoms never carry an ECP
            (default: {None})
        spin {int} -- Number of unpaired electrons (default: {0})
        verbose {int} -- PySCF logger level (default: {VERBOSE})
        output {str | None} -- Log file for this molecule; None logs to
            stdout (default: {None})
        cart {bool} -- Use Cartesian d/f functions (default: {False})

    Raises:
        ValueError -- If no basis set is given for an element.

    Returns:
        {pyscf.gto.Mole} -- The built molecule.
    """
    spec, bas, ecp_real = [], {}, {}
    for i, (label, xyz) in enumerate(atoms):
        el = _element(label)
        name = _name_for(basis, el)
        if name is None:
            raise ValueError(f'no basis set given for {el}')
        bas[label] = name
        bas['ghost-' + label] = gto.basis.load(name, el)
        if i in ghost_idx:
            spec.append(['ghost-' + label, tuple(xyz)])
        else:
            spec.append([label, tuple(xyz)])
            if ecp and el in ecp:
                ecp_real[label] = ecp[el]
    return gto.M(atom=spec, basis=bas, ecp=ecp_real, spin=spin,
                 charge=charge, unit='Angstrom', verbose=verbose,
                 output=output, cart=cart)


def _df_auxbasis(mol: gto.Mole, auxname: str | None) -> dict | None:
    """Build an auxiliary basis with the same functions on ghost atoms.

    All sobEDA stages then share one fitting space, so density-fitting
    errors largely cancel in the differences.

    Arguments:
        mol {pyscf.gto.Mole} -- Molecule, possibly with ghost atoms.
        auxname {str | None} -- Auxiliary basis name; None for the PySCF
            default.

    Returns:
        {dict | None} -- Auxiliary basis per atom label, or None.
    """
    if auxname is None:
        return None
    aux = {}
    for key in mol._basis:
        if key.lower().startswith('ghost-'):
            aux[key] = gto.basis.load(auxname, _element(key[6:]))
        else:
            aux[key] = auxname
    return aux


def _d3bj(mol: gto.Mole, xc: str, disp: str) -> float:
    """Compute the D3 or D4 dispersion energy.

    Arguments:
        mol {pyscf.gto.Mole} -- Molecule.
        xc {str} -- Functional; selects the damping parameters.
        disp {str} -- 'd3bj', 'd3zero' or 'd4'.

    Returns:
        {float} -- Dispersion energy in Hartree.
    """
    from pyscf.dispersion import dftd3, dftd4
    if disp.lower() == 'd4':
        d = dftd4.DFTD4Dispersion(mol, xc=xc.upper())
    else:
        d = dftd3.DFTD3Dispersion(mol, xc=xc.upper(), version=disp.lower())
    return float(d.get_dispersion()['energy'])


def _effective_disp(xc: str, disp: str | None) -> str | None:
    """Return the dispersion correction that is actually applied.

    VV10 functionals already describe dispersion, so an additional D3/D4
    correction would double count it and is dropped.

    Arguments:
        xc {str} -- Functional.
        disp {str | None} -- Requested dispersion correction.

    Returns:
        {str | None} -- disp, or None for VV10 functionals.
    """
    if disp is not None and _libxc.is_nlc(xc):
        return None
    return disp


def _chk_matches(chkfile: str, mol: gto.Mole) -> bool:
    """Check whether a checkpoint file belongs to this molecule and basis.

    Arguments:
        chkfile {str} -- Path to a PySCF checkpoint file.
        mol {pyscf.gto.Mole} -- Current molecule.

    Returns:
        {bool} -- True if atoms, ghost pattern, coordinates, charge and
            basis size agree.
    """
    try:
        old = _chkfile.load_mol(chkfile)
    except Exception:
        return False
    return (old.natm == mol.natm and old.nao == mol.nao
            and old.charge == mol.charge
            and [old.atom_symbol(i) for i in range(old.natm)]
            == [mol.atom_symbol(i) for i in range(mol.natm)]
            and np.allclose(old.atom_coords(), mol.atom_coords(), atol=1e-6))


def _make_ks(mol: gto.Mole, xc: str, grid_level: int = GRID_LEVEL,
             chkfile: str | None = None,
             use_df: bool | str = False) -> RKS:
    """Set up an RKS object with the module's grid, SCF and DF settings.

    Arguments:
        mol {pyscf.gto.Mole} -- Molecule.
        xc {str} -- Functional.

    Keyword Arguments:
        grid_level {int} -- PySCF DFT grid level (default: {GRID_LEVEL})
        chkfile {str | None} -- Checkpoint file; an existing one for the
            same molecule and basis is used as the initial guess
            (default: {None})
        use_df {bool | str} -- False, True (RI-JK) or 'j-only' (RI-J)
            (default: {False})

    Returns:
        {pyscf.dft.rks.RKS} -- KS object (pyscf or gpu4pyscf), not yet run.
    """
    mf = _dft_mod.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    if _libxc.is_nlc(xc):
        mf.nlcgrids.level = NLC_GRID_LEVEL   # VV10 itself is automatic
    mf.max_cycle = MAX_SCF_CYCLE
    mf.conv_tol = CONV_TOL
    if use_df:
        aux = _df_auxbasis(mol, DF_AUXBASIS)
        # Dense (naux, nao(nao+1)/2) upper bound; gpu4pyscf keeps the tensor
        # in GPU/host-mapped memory, so an oversized one is a hard OOM.
        from pyscf.df import addons as _df_addons
        _auxmol = _df_addons.make_auxmol(mol, aux)
        _nao, _naux = mol.nao, _auxmol.nao
        _est_gb = _naux * (_nao * (_nao + 1) / 2) * 8 / 1e9
        logger.info(mol, 'DF tensor (dense) estimate: nao=%d naux=%d '
                    '-> %.1f GB', _nao, _naux, _est_gb)
        if _est_gb > DF_CDERI_LIMIT_GB:
            logger.warn(mol,
                        'density fitting DISABLED: estimated DF tensor '
                        '%.0f GB exceeds DF_CDERI_LIMIT_GB=%g GB. Falling '
                        'back to integral-direct J/K.',
                        _est_gb, DF_CDERI_LIMIT_GB)
            use_df = False
    if use_df == 'j-only':
        if 'only_dfj' in inspect.signature(mf.density_fit).parameters:
            mf = mf.density_fit(auxbasis=aux, only_dfj=True)
        else:
            # RI-JK instead would hit the memory problem the guard prevents.
            logger.warn(mol, 'this backend has no only_dfj (RI-J with '
                        'exact K); proceeding WITHOUT density fitting.')
    elif use_df:
        mf = mf.density_fit(auxbasis=aux)
    if chkfile:
        mf.chkfile = chkfile
        if os.path.exists(chkfile):
            if _chk_matches(chkfile, mol):
                mf.init_guess = 'chkfile'
                logger.note(mol, 'initial guess from %s', chkfile)
            else:
                logger.warn(mol, '%s belongs to another molecule or basis; '
                            'ignored and overwritten', chkfile)
    return mf


def _exact_exchange(mf: RKS, dm: np.ndarray | cupy.ndarray,
                    xc: str) -> float:
    """Compute the exact-exchange energy, including range separation.

    Mirrors pyscf.dft.rks.get_veff:

        omega == 0 : K_tot = hyb K
        alpha == 0 : K_tot = hyb K_SR(omega)
        hyb == 0   : K_tot = alpha K_LR(omega)
        otherwise  : K_tot = hyb K + (alpha - hyb) K_LR(omega)

    with E_x^HF = -1/4 Tr[K_tot P].

    Arguments:
        mf {pyscf.dft.rks.RKS} -- KS object providing get_k.
        dm {numpy.ndarray | cupy.ndarray} -- Closed-shell density matrix P.
        xc {str} -- Functional; sets hyb, alpha and omega.

    Returns:
        {float} -- Exact-exchange energy in Hartree (0 for pure DFT).
    """
    mol, ni = mf.mol, mf._numint
    dm_np = _to_np(dm)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
    if omega == 0:
        if abs(hyb) < 1e-14:
            return 0.0
        vk = hyb * _to_np(mf.get_k(mol, dm))
    elif abs(alpha) < 1e-14:
        vk = hyb * _to_np(mf.get_k(mol, dm, omega=-omega))
    elif abs(hyb) < 1e-14:
        vk = alpha * _to_np(mf.get_k(mol, dm, omega=omega))
    else:
        vk = (hyb * _to_np(mf.get_k(mol, dm))
              + (alpha - hyb) * _to_np(mf.get_k(mol, dm, omega=omega)))
    return -0.25 * float(np.einsum('ij,ji->', vk, dm_np).real)


def _components(mf: RKS, dm: np.ndarray, xc: str,
                disp: str | None = None) -> dict:
    """Compute the energy components of a density.

    Uses mf's own J/K builders, so density fitting and the GPU backend are
    respected and E_tot matches mf.e_tot. As in sobEDA.sh, E_els collects
    everything except kinetic, exchange-correlation and dispersion energy,
    so it includes the ECP energy.

    Arguments:
        mf {pyscf.dft.rks.RKS} -- KS object whose basis and grids match dm.
        dm {numpy.ndarray} -- Closed-shell density matrix.
        xc {str} -- Functional.

    Keyword Arguments:
        disp {str | None} -- 'd3bj', 'd3zero' or 'd4'; None adds no
            dispersion correction (default: {None})

    Returns:
        {dict} -- Energies in Hartree: E_tot, T, E_els, E_disp, plus E_x
            and E_c for functionals in XC_SPLIT, or otherwise E_xc and its
            parts E_xc_sl (semilocal), E_xc_nlc (VV10), E_xc_hfx (exact).
    """
    mol, ni, grids = mf.mol, mf._numint, mf.grids
    dm = _to_np(dm)          # host copy for CPU integrals
    dm_d = _to_dev(dm)       # device copy for gpu4pyscf
    kin = mol.intor('int1e_kin')
    vne = mol.intor('int1e_nuc')
    if mol.has_ecp():
        vne = vne + mol.intor('ECPscalar')
    vj = _to_np(mf.get_j(mol, dm_d))

    T = float(np.einsum('ij,ji->', kin, dm).real)
    E_ne = float(np.einsum('ij,ji->', vne, dm).real)
    E_J = 0.5 * float(np.einsum('ij,ji->', vj, dm).real)
    E_nn = float(mol.energy_nuc())
    E_els = E_nn + E_ne + E_J

    E_x_hf = _exact_exchange(mf, dm_d, xc)
    E_disp = _d3bj(mol, xc, disp) if disp is not None else 0.0
    core = E_nn + float(np.einsum('ij,ji->', kin + vne, dm).real) + E_J

    if xc.upper() in XC_SPLIT:
        xstr, cstr = XC_SPLIT[xc.upper()]
        E_x_dft = float(_to_np(ni.nr_rks(mol, grids, xstr, dm_d)[1]))
        E_c = float(_to_np(ni.nr_rks(mol, grids, cstr, dm_d)[1]))
        E_x = E_x_dft + E_x_hf
        E_tot = core + E_x + E_c + E_disp
        return dict(E_tot=E_tot, T=T, E_els=E_els,
                    E_x=E_x, E_c=E_c, E_disp=E_disp)

    # Inseparable functional: one combined XC term.
    E_xc_sl = float(_to_np(ni.nr_rks(mol, grids, xc, dm_d)[1]))
    E_xc_nlc = 0.0
    if _libxc.is_nlc(xc):
        if mf.nlcgrids.coords is None:
            mf.nlcgrids.build()
        E_xc_nlc = float(_to_np(
            ni.nr_nlc_vxc(mol, mf.nlcgrids, xc, dm_d)[1]))
    E_xc = E_xc_sl + E_xc_nlc + E_x_hf
    E_tot = core + E_xc + E_disp
    return dict(E_tot=E_tot, T=T, E_els=E_els,
                E_xc=E_xc, E_xc_sl=E_xc_sl, E_xc_nlc=E_xc_nlc,
                E_xc_hfx=E_x_hf, E_disp=E_disp)


def _frozen_density(C_occ: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Build the frozen-state density from the occupied fragment MOs.

    Arguments:
        C_occ {numpy.ndarray} -- Occupied MO coefficients of all fragments,
            shape (nao, nocc).
        S {numpy.ndarray} -- AO overlap matrix.

    Raises:
        RuntimeError -- If the occupied MOs are (nearly) linearly dependent,
            e.g. because fragments overlap.

    Returns:
        {numpy.ndarray} -- Closed-shell density 2 C C^T of the
            Loewdin-orthonormalised occupied MOs C.
    """
    w, U = np.linalg.eigh(C_occ.T @ S @ C_occ)
    if w.min() < LINDEP_TOL:
        raise RuntimeError(f'occupied fragment orbitals are linearly '
                           f'dependent (smallest overlap eigenvalue '
                           f'{w.min():.1e}); do fragments overlap?')
    C = C_occ @ (U @ np.diag(w ** -0.5) @ U.T)
    return 2.0 * C @ C.T


def _save_state(mol: gto.Mole, tag: str, outdir: str,
                mo_coeff: np.ndarray | None = None,
                mo_energy: np.ndarray | None = None,
                mo_occ: np.ndarray | None = None,
                dm: np.ndarray | None = None,
                log: logger.Logger | None = None) -> None:
    """Write Molden (if MOs are given) and density cube files for a stage.

    Arguments:
        mol {pyscf.gto.Mole} -- Molecule of the stage.
        tag {str} -- File name stem, e.g. 'frag1' or 'final'.
        outdir {str} -- Output directory, created if missing.

    Keyword Arguments:
        mo_coeff {numpy.ndarray | None} -- MO coefficients
            (default: {None})
        mo_energy {numpy.ndarray | None} -- MO energies (default: {None})
        mo_occ {numpy.ndarray | None} -- MO occupations (default: {None})
        dm {numpy.ndarray | None} -- Density matrix for the cube file
            (default: {None})
        log {pyscf.lib.logger.Logger | None} -- Logger for the written file
            names (default: {None})
    """
    os.makedirs(outdir, exist_ok=True)
    if mo_coeff is not None:
        fname = os.path.join(outdir, f'{tag}.molden')
        molden.from_mo(mol, fname, mo_coeff, ene=mo_energy, occ=mo_occ)
        if log:
            log.info('  saved %s', fname)
    if dm is not None:
        fname = os.path.join(outdir, f'{tag}_density.cube')
        cubegen.density(mol, fname, dm)
        if log:
            log.info('  saved %s', fname)


def _eda_terms(frag_E_list: list[dict], promol: dict, frozen: dict,
               final: dict) -> dict:
    """Assemble the sobEDA terms from the stage energy components.

    Works for the full system or a fragment subset (pair), and for six-term
    and five-term functionals.

    Arguments:
        frag_E_list {list[dict]} -- Components of each isolated fragment.
        promol {dict} -- Components of the promolecule.
        frozen {dict} -- Components of the frozen state.
        final {dict} -- Components of the final state.

    Returns:
        {dict} -- sobEDA terms in Hartree.
    """
    def dsum(key: str) -> float:
        return promol[key] - sum(f[key] for f in frag_E_list)

    E_rep = frozen['E_tot'] - promol['E_tot']
    E_orb = final['E_tot'] - frozen['E_tot']
    E_int = final['E_tot'] - sum(f['E_tot'] for f in frag_E_list)
    terms = dict(E_els=dsum('E_els'), E_rep=E_rep, E_orb=E_orb,
                 E_int=E_int, E_dc=dsum('E_disp'))

    if 'E_x' in promol:
        terms['E_x'] = dsum('E_x')
        terms['E_DFTc'] = dsum('E_c')
        terms['E_xrep'] = terms['E_x'] + E_rep
        terms['E_c'] = terms['E_DFTc'] + terms['E_dc']
    else:
        terms['E_xc'] = dsum('E_xc')
        terms['E_xc_sl'] = dsum('E_xc_sl')
        terms['E_xc_nlc'] = dsum('E_xc_nlc')
        terms['E_xc_hfx'] = dsum('E_xc_hfx')
        terms['E_xcrep'] = terms['E_xc'] + E_rep
    return terms


def _sum_check(terms: dict) -> float:
    """Compute the closure error of a decomposition.

    Arguments:
        terms {dict} -- sobEDA terms in Hartree.

    Returns:
        {float} -- Sum of the terms minus E_int in Hartree (zero by
            construction; guards against bookkeeping errors).
    """
    stage1 = terms['E_els'] + terms['E_dc'] + (
        terms['E_x'] + terms['E_DFTc'] if 'E_x' in terms else terms['E_xc'])
    return stage1 + terms['E_rep'] + terms['E_orb'] - terms['E_int']


def _check_inputs(natom: int, fragments: list[dict], xc: str,
                  disp: str | None, use_df: bool | str,
                  sobedaw_parm: tuple | None) -> None:
    """Validate the arguments of sobeda that need no calculation.

    Arguments:
        natom {int} -- Number of atoms in the system.
        fragments {list[dict]} -- Fragment definitions.
        xc {str} -- Functional (upper case).
        disp {str | None} -- Dispersion correction.
        use_df {bool | str} -- Density fitting setting.
        sobedaw_parm {tuple | None} -- sobEDAw parameters (c, a, r).

    Raises:
        ValueError -- On the first invalid argument; atom and fragment
            numbers in the message are 1-based.
    """
    if len(fragments) < 2:
        raise ValueError('at least two fragments are required')
    for k, f in enumerate(fragments, 1):
        if 'atoms' not in f or 'charge' not in f:
            raise ValueError(f"fragment {k}: 'atoms' and 'charge' are "
                             f"required")
        if f.get('mult', 1) != 1:
            raise ValueError(f'fragment {k}: only closed-shell (singlet) '
                             f'fragments are supported (RKS)')
    counts = Counter(i for f in fragments for i in f['atoms'])
    missing = [i + 1 for i in range(natom) if i not in counts]
    shared = sorted(i + 1 for i, n in counts.items()
                    if n > 1 and 0 <= i < natom)
    outside = sorted(i + 1 for i in counts if not 0 <= i < natom)
    if missing or shared or outside:
        raise ValueError(f'fragments must cover each of the {natom} atoms '
                         f'exactly once (1-based atom numbers): missing '
                         f'{missing}, in several fragments {shared}, out '
                         f'of range {outside}')
    if disp is not None and disp.lower() not in DISP_VERSIONS:
        raise ValueError(f'disp must be one of {DISP_VERSIONS} or None, '
                         f'got {disp!r}')
    if use_df not in (False, True, 'j-only'):
        raise ValueError(f"use_df must be False, True or 'j-only', got "
                         f"{use_df!r}")
    if sobedaw_parm is not None:
        if len(sobedaw_parm) != 3:
            raise ValueError('sobedaw_parm must be (c, a, r)')
        if xc not in XC_SPLIT:
            raise ValueError(f'sobEDAw is not defined for {xc} (no E_DFTc '
                             f'/ E_dc split). Separable functionals: '
                             f'{", ".join(XC_SPLIT)}.')


def sobeda(atoms: list[tuple], fragments: list[dict], xc: str,
           basis: str | dict, disp: str | None = None,
           ecp: str | dict | None = None, grid_level: int = GRID_LEVEL,
           save_outputs: bool = False, save_cubes: bool = False,
           outdir: str = 'output', verbose: int = VERBOSE,
           logfile: str | None = None, pairwise: bool = True,
           use_df: bool | str = False, cart: bool = False,
           sobedaw_parm: tuple[float, float, float] | None = None,
           save_chk: bool = True, save_molden: bool = True,
           ) -> tuple[dict, dict, dict | None]:
    """Run a sobEDA energy decomposition.

    Arguments:
        atoms {list[tuple]} -- (element, (x, y, z)) for each atom in
            Angstrom, as returned by read_xyz.
        fragments {list[dict]} -- Fragment definitions as returned by
            read_fragments; 'mult' must be 1. The total charge is the sum
            of the fragment charges.
        xc {str} -- Functional. Six-term sobEDA if listed in XC_SPLIT,
            otherwise five-term with a combined E_xc.
        basis {str | dict} -- Orbital basis set (PySCF name), or
            {element: name} with an optional 'default' entry.

    Keyword Arguments:
        disp {str | None} -- 'd3bj', 'd3zero' or 'd4'; dropped, with a
            warning, for VV10 functionals (default: {None})
        ecp {str | dict | None} -- ECP name, or {element: name} with an
            optional 'default' entry. Elements not covered get the ECP of
            their basis set, if PySCF has one (default: {None})
        grid_level {int} -- PySCF DFT grid level (default: {GRID_LEVEL})
        save_outputs {bool} -- Write a JSON summary to outdir, plus chk and
            Molden files for every stage unless switched off by save_chk
            and save_molden (default: {False})
        save_cubes {bool} -- With save_outputs, also write a density cube
            file for every stage (default: {False})
        outdir {str} -- Output directory (default: {'output'})
        verbose {int} -- PySCF logger level (default: {VERBOSE})
        logfile {str | None} -- Write progress messages to this file
            instead of stdout; with save_outputs each SCF also gets its own
            log file in outdir (default: {None})
        save_chk {bool} -- With save_outputs, keep a checkpoint file per
            SCF; an existing one for the same molecule is the initial
            guess (default: {True})
        save_molden {bool} -- With save_outputs, write a Molden file for
            every stage with canonical MOs (default: {True})
        pairwise {bool} -- For more than two fragments, also decompose
            every fragment pair and report the many-body remainder
            (default: {True})
        use_df {bool | str} -- Density fitting: False, True (RI-JK) or
            'j-only' (RI-J with exact exchange) (default: {False})
        cart {bool} -- Cartesian d/f functions; matches Gaussian's 6D
            default for the 6-31G family if no f functions are present
            (default: {False})
        sobedaw_parm {tuple[float, float, float] | None} -- sobEDAw
            parameters (c, a, r); validated before any SCF and added to the
            JSON summary (default: {None})

    Raises:
        ValueError -- If an input is invalid: fewer than two fragments,
            fragments not covering every atom exactly once, an open-shell
            or odd-electron fragment, an unknown disp or use_df, an
            unavailable explicit ECP, or sobEDAw for a five-term
            functional. All are checked before any SCF.
        RuntimeError -- If an SCF does not converge or the fragments'
            occupied orbitals are linearly dependent.

    Returns:
        {tuple[dict, dict, dict | None]} -- (terms, checks, pairs): the
            sobEDA terms in Hartree; the checks recon_max_abs (component
            reconstruction vs SCF energy), dT1 and sum_minus_int (all ~0),
            recon_Ha per stage, the fragment energies and the total charge;
            and, if the pairwise decomposition ran,
            {'pairs': {(i, j): terms}, 'many_body': terms} with 0-based
            fragment indices, else None.
    """
    xc = _canonical_xc(xc)
    natom = len(atoms)
    allidx = set(range(natom))
    _check_inputs(natom, fragments, xc, disp, use_df, sobedaw_parm)
    ecp_map = _ecp_map(atoms, basis, ecp)
    for k, f in enumerate(fragments, 1):
        nel = _n_electrons(atoms, f['atoms'], f['charge'], ecp_map)
        if nel % 2:
            raise ValueError(f'fragment {k} has {nel} electrons; only '
                             f'closed-shell fragments are supported')

    if save_outputs:
        os.makedirs(outdir, exist_ok=True)
    # pyscf's Logger needs a real stream, never None.
    log_fh = open(logfile, 'a') if logfile else None
    log = logger.Logger(log_fh or sys.stdout, verbose)
    try:
        if _effective_disp(xc, disp) != disp:
            log.warn('%s already contains VV10 nonlocal correlation; the %s '
                     'correction is not applied (double counting).', xc, disp)
            disp = None
        do_disp = disp is not None

        separable = xc in XC_SPLIT
        log.note('Functional %s: %s', xc,
                 'separable exchange/correlation -> six-term sobEDA.'
                 if separable else
                 'combined (inseparable) XC -> five-term sobEDA (E_xc reported '
                 'in place of E_x/E_DFTc; sobEDAw not applicable).')
        if ecp_map:
            log.note('ECP: %s', _spec_str(ecp_map))
        if sobedaw_parm is not None:
            if not do_disp:
                log.warn('sobEDAw requested without a dispersion correction: '
                         'E_dc = 0, so w is not meaningful. The fitted (c, a, r) '
                         'parameters assume DFT-D3(BJ).')
            if xc not in SOBEDAW_FUNCTIONALS:
                log.warn('no fitted sobEDAw parameters exist for %s in the '
                         'sobEDA paper (fitted: %s); functionals that partly '
                         'describe dispersion are incompatible with sobEDAw.',
                         xc, ', '.join(SOBEDAW_FUNCTIONALS))

        total_charge = sum(f['charge'] for f in fragments)
        recon = {}    # E_tot(reconstructed) - E_tot(SCF)
        per_scf_log = save_outputs and logfile

        def stage_mol(ghosts: set[int], charge: int, tag: str) -> gto.Mole:
            out = (os.path.join(outdir, f'{tag}.pyscf_out')
                   if per_scf_log else None)
            return _mol(atoms, ghosts, basis, charge, ecp=ecp_map,
                        verbose=verbose, output=out, cart=cart)

        def stage_ks(mol: gto.Mole, tag: str) -> RKS:
            chk = (os.path.join(outdir, f'{tag}.chk')
                   if save_outputs and save_chk else None)
            mf = _make_ks(mol, xc, grid_level, chkfile=chk, use_df=use_df)
            if do_disp:
                mf.disp = disp
            return mf

        # Fragments, each in the full basis with the other atoms as ghosts.
        frag_dm, frag_moocc, frag_E = [], [], []
        for i, f in enumerate(fragments):
            tag = f'frag{i + 1}'
            log.note('--- Fragment %d (charge %+d) SCF ---', i + 1, f['charge'])
            mf = stage_ks(stage_mol(allidx - set(f['atoms']), f['charge'], tag),
                          tag)
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f'fragment {i + 1} SCF not converged')
            log.note('  E_tot(%s) = %.8f Ha', tag, mf.e_tot)

            dm = _to_np(mf.make_rdm1())
            frag_dm.append(dm)
            mo, occ = _to_np(mf.mo_coeff), _to_np(mf.mo_occ)
            mo_energy = _to_np(mf.mo_energy)
            frag_moocc.append(mo[:, occ > 0])
            comps = _components(mf, dm, xc, disp)
            frag_E.append(comps)
            recon[tag] = comps['E_tot'] - float(mf.e_tot)

            if save_outputs:
                _save_state(mf.mol, tag, outdir,
                            mo_coeff=mo if save_molden else None,
                            mo_energy=mo_energy, mo_occ=occ,
                            dm=dm if save_cubes else None, log=log)
            mf = None    # free DF tensors / GPU memory before the next SCF

        # Supermolecule.
        log.note('--- Supermolecule (final state) SCF ---')
        mf_full = stage_ks(stage_mol(set(), total_charge, 'final'), 'final')
        mol = mf_full.mol
        S = _to_np(mf_full.get_ovlp())

        # Frozen state; it is also the initial guess of the supermolecule SCF.
        P_froz = _frozen_density(np.hstack(frag_moocc), S)
        if mf_full.init_guess == 'chkfile':
            mf_full.kernel()
        else:
            mf_full.kernel(dm0=_to_dev(P_froz))
        if not mf_full.converged:
            raise RuntimeError('supermolecule SCF not converged')
        log.note('  E_tot(final) = %.8f Ha', mf_full.e_tot)

        # Promolecule: sum of the fragment densities.
        P_promol = sum(frag_dm)
        promol = _components(mf_full, P_promol, xc, disp)
        log.note('  E_tot(promolecule) = %.8f Ha', promol['E_tot'])

        frozen = _components(mf_full, P_froz, xc, disp)
        log.note('  E_tot(frozen state) = %.8f Ha', frozen['E_tot'])

        final = _components(mf_full, _to_np(mf_full.make_rdm1()), xc,
                            disp)
        recon['final'] = final['E_tot'] - float(mf_full.e_tot)

        if save_outputs:
            _save_state(mol, 'final', outdir,
                        mo_coeff=(_to_np(mf_full.mo_coeff)
                                  if save_molden else None),
                        mo_energy=_to_np(mf_full.mo_energy),
                        mo_occ=_to_np(mf_full.mo_occ),
                        dm=_to_np(mf_full.make_rdm1()) if save_cubes else None,
                        log=log)
            if save_cubes:    # no canonical MOs for these two, only densities
                _save_state(mol, 'promolecule', outdir, dm=P_promol, log=log)
                _save_state(mol, 'frozen', outdir, dm=P_froz, log=log)

        terms = _eda_terms(frag_E, promol, frozen, final)

        checks = dict(
            recon_max_abs=max(abs(v) for v in recon.values()),
            dT1=promol['T'] - sum(f['T'] for f in frag_E),
            sum_minus_int=_sum_check(terms),
            recon_Ha=recon,
            frag_Etot=[f['E_tot'] for f in frag_E],
            total_charge=total_charge,
        )
        if checks['recon_max_abs'] > 1e-5:
            log.warn('component reconstruction differs from SCF e_tot by '
                     '%.2e Ha -- the E_x/E_c (or E_xc) split may be '
                     'inconsistent with the SCF functional.',
                     checks['recon_max_abs'])

        # Pairwise decomposition: one extra SCF per pair. All stages share the
        # full basis, so the fragment densities and MOs are reused as they are.
        pairs: dict | None = None
        if pairwise and len(fragments) > 2:
            pair_terms: dict = {}
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    fi, fj = fragments[i], fragments[j]
                    tag = f'pair{i + 1}-{j + 1}'
                    log.note('--- Pair %d-%d (charges %+d, %+d) SCF ---',
                             i + 1, j + 1, fi['charge'], fj['charge'])
                    ghosts = allidx - set(fi['atoms']) - set(fj['atoms'])
                    mf_ij = stage_ks(stage_mol(ghosts, fi['charge'] + fj['charge'],
                                               tag), tag)
                    # Frozen state of the pair, also its initial guess.
                    P_froz_ij = _frozen_density(
                        np.hstack((frag_moocc[i], frag_moocc[j])), S)
                    if mf_ij.init_guess == 'chkfile':
                        mf_ij.kernel()
                    else:
                        mf_ij.kernel(dm0=_to_dev(P_froz_ij))
                    if not mf_ij.converged:
                        raise RuntimeError(f'pair {i + 1}-{j + 1} SCF not '
                                           f'converged')
                    log.note('  E_tot(%s) = %.8f Ha', tag, mf_ij.e_tot)

                    promol_ij = _components(mf_ij, frag_dm[i] + frag_dm[j],
                                            xc, disp)
                    frozen_ij = _components(mf_ij, P_froz_ij,
                                            xc, disp)
                    final_ij = _components(mf_ij, _to_np(mf_ij.make_rdm1()),
                                           xc, disp)
                    pair_terms[(i, j)] = _eda_terms(
                        [frag_E[i], frag_E[j]], promol_ij, frozen_ij, final_ij)
                    mf_ij = None

            pairs = {'pairs': pair_terms,
                     'many_body': {k: terms[k] - sum(t[k] for t in
                                                     pair_terms.values())
                                   for k in terms}}

        if save_outputs:
            level = (f'{xc}{"-" + disp.upper() if disp else ""}'
                     f'/{_spec_str(basis)}'
                     f'{" (cart)" if cart else ""}'
                     f'{" ECP: " + _spec_str(ecp_map) if ecp_map else ""}'
                     f'{" (RI-JK)" if use_df is True else ""}'
                     f'{" (RI-J)" if use_df == "j-only" else ""}')
            summary = {
                'level_of_theory': level,
                'terms_kcal_per_mol': {k: v * H2KCAL for k, v in terms.items()},
                'checks': checks,
                'stage_E_tot_Ha': {
                    **{f'frag{i + 1}': f['E_tot'] for i, f in enumerate(frag_E)},
                    'promolecule': promol['E_tot'],
                    'frozen': frozen['E_tot'],
                    'final': final['E_tot'],
                },
            }
            if sobedaw_parm is not None:
                tw = sobedaw(terms, *sobedaw_parm)
                summary['sobEDAw'] = {
                    'c_a_r': list(sobedaw_parm), 'w': tw['w'],
                    'w_unclipped': tw['w_unclipped'],
                    'terms_kcal_per_mol': {k: tw[k] * H2KCAL for k in
                                           ('E_els', 'E_xrep', 'E_orb', 'E_disp')},
                }
            if pairs:
                summary['pairwise_kcal_per_mol'] = {
                    f'{i + 1}-{j + 1}': {k: v * H2KCAL for k, v in t.items()}
                    for (i, j), t in pairs['pairs'].items()}
                summary['many_body_kcal_per_mol'] = {
                    k: v * H2KCAL for k, v in pairs['many_body'].items()}
            fname = os.path.join(outdir, 'sobeda_summary.json')
            with open(fname, 'w') as fh:
                json.dump(summary, fh, indent=2)
            log.note('  saved %s', fname)

        return terms, checks, pairs
    finally:
        if log_fh:
            log_fh.close()


def sobedaw(terms: dict, c: float, a: float, r: float) -> dict:
    """Apply the sobEDAw scheme to six-term sobEDA terms.

    Same as eqs 3-5 of the sobEDA paper and sobEDA.sh:

        w = min(1, exp(-r (E_dc / E_els - a)) (1 - c) + c)
        E_disp = E_dc + w E_DFTc
        E_xrep = E_x + E_rep + (1 - w) E_DFTc

    Arguments:
        terms {dict} -- sobEDA terms returned by sobeda for a functional in
            XC_SPLIT.
        c {float} -- Fitted parameter c (Tables 1-2 of the sobEDA paper).
        a {float} -- Fitted parameter a.
        r {float} -- Fitted parameter r.

    Raises:
        ValueError -- If terms has no E_DFTc (five-term result).

    Returns:
        {dict} -- w, w_unclipped and the terms E_els, E_xrep, E_orb and
            E_disp in Hartree.
    """
    if 'E_DFTc' not in terms:
        raise ValueError(
            'sobEDAw is not defined for inseparable/NLC functionals '
            '(no E_DFTc / E_dc split exists). Use a separable '
            'D3-corrected level such as BLYP-D3(BJ) or B3LYP-D3(BJ).')
    if terms['E_els'] == 0:
        raise ValueError('sobEDAw needs E_els != 0 (w depends on E_dc/E_els)')
    ratio = terms['E_dc'] / terms['E_els']
    w_unclipped = float(np.exp(-r * (ratio - a)) * (1 - c) + c)
    w = min(w_unclipped, 1.0)
    return dict(
        w=w,
        w_unclipped=w_unclipped,
        E_els=terms['E_els'],
        E_xrep=terms['E_xrep'] + (1 - w) * terms['E_DFTc'],
        E_orb=terms['E_orb'],
        E_disp=terms['E_dc'] + w * terms['E_DFTc'],
    )


_LABELS = [
    ('E_int',    'ΔE_int (total interaction)'),
    ('E_els',    'ΔE_els (electrostatic)'),
    ('E_x',      'ΔE_x (exchange)'),
    ('E_xc',     'ΔE_xc (exchange-correlation)'),
    ('E_xc_sl',  '  ΔE_xc semilocal part'),
    ('E_xc_nlc', '  ΔE_xc VV10 nonlocal part'),
    ('E_xc_hfx', '  ΔE_xc exact-exchange part'),
    ('E_rep',    'ΔE_rep (Pauli repulsion)'),
    ('E_xrep',   'ΔE_xrep (= x + rep)'),
    ('E_xcrep',  'ΔE_xcrep (= xc + rep)'),
    ('E_orb',    'ΔE_orb (orbital relaxation)'),
    ('E_DFTc',   'ΔE_DFTc (DFT correlation)'),
    ('E_dc',     'ΔE_dc (dispersion correction)'),
]

_LABELS_W = [
    ('E_els',  'ΔE_els (electrostatic)'),
    ('E_xrep', 'ΔE_xrep (incl. scaled DFT corr.)'),
    ('E_orb',  'ΔE_orb (orbital relaxation)'),
    ('E_disp', 'ΔE_disp (dispersion)'),
]


def print_results(terms: dict, checks: dict, pairs: dict | None = None,
                  terms_w: dict | None = None) -> None:
    """Print the sobEDA, sobEDAw and pairwise terms in kcal/mol.

    Arguments:
        terms {dict} -- sobEDA terms returned by sobeda.
        checks {dict} -- Consistency checks returned by sobeda.

    Keyword Arguments:
        pairs {dict | None} -- Pairwise terms returned by sobeda
            (default: {None})
        terms_w {dict | None} -- sobEDAw terms returned by sobedaw
            (default: {None})
    """
    print(f" {'Total charge of supersystem':<34}{checks['total_charge']}")
    for key, label in _LABELS:
        if key in terms:
            unit = ' kcal/mol' if key == 'E_int' else ''
            print(f" {label:<34}{terms[key]*H2KCAL:9.2f}{unit}")
    if terms_w is not None:
        print(" --- sobEDAw (kcal/mol) ---")
        if terms_w['w_unclipped'] > 1.0:
            print(f"   Warning: w = {terms_w['w_unclipped']:.4f} > 1, "
                  f"scaled to 1.0")
        print(f"   {terms_w['w']*100:.2f}% of ΔE_DFTc combined with ΔE_dc "
              f"into a SAPT-like dispersion term")
        for key, label in _LABELS_W:
            print(f" {label:<34}{terms_w[key]*H2KCAL:9.2f}")
    print(" --- consistency checks (a.u., should be ~0) ---")
    print(f"   max |E_recon - E_scf|:        {checks['recon_max_abs']:.2e}")
    print(f"   ΔT_1, Σterms - ΔE_int:        {checks['dT1']:.2e}, "
          f"{checks['sum_minus_int']:.2e}  (zero by construction)")

    if pairs:
        keys = [k for k, _ in _LABELS
                if k in terms and not k.startswith('E_xc_')
                and k not in ('E_xrep', 'E_xcrep')]
        print("\n --- pairwise decomposition (kcal/mol) ---")
        print(f" {'pair':<10}" + ''.join(f'{k:>9}' for k in keys))
        for (i, j), t in sorted(pairs['pairs'].items()):
            label = f'{i + 1}-{j + 1}'
            print(f" {label:<10}"
                  + ''.join(f'{t[k]*H2KCAL:9.2f}' for k in keys))
        pair_sum = {k: sum(t[k] for t in pairs['pairs'].values())
                    for k in keys}
        print(f" {'Σ pairs':<10}"
              + ''.join(f'{pair_sum[k]*H2KCAL:9.2f}' for k in keys))
        print(f" {'many-body':<10}"
              + ''.join(f'{pairs["many_body"][k]*H2KCAL:9.2f}'
                        for k in keys))
        print(f" {'total':<10}"
              + ''.join(f'{terms[k]*H2KCAL:9.2f}' for k in keys))


def _parse_spec(tokens: list[str] | None) -> str | dict | None:
    """Turn command-line tokens 'NAME' and 'El=NAME' into a specification.

    Arguments:
        tokens {list[str] | None} -- Tokens; at most one without '='.

    Raises:
        ValueError -- If a token is empty or an element is given twice.

    Returns:
        {str | dict | None} -- The name if only 'NAME' is given, else
            {element: name} with an optional 'default' entry.
    """
    if tokens is None:
        return None
    spec = {}
    for tok in tokens:
        key, sep, name = tok.partition('=')
        key = _element(key) if sep else 'default'
        name = name if sep else tok
        if not name or key in spec:
            raise ValueError(f'invalid or repeated entry {tok!r}')
        spec[key] = name
    return spec['default'] if list(spec) == ['default'] else spec


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Keyword Arguments:
        argv {list[str] | None} -- Arguments; None reads sys.argv
            (default: {None})

    Returns:
        {argparse.Namespace} -- Parsed arguments; basis and ecp are already
            converted by _parse_spec.
    """
    p = argparse.ArgumentParser(
        description='sobEDA / sobEDAw energy decomposition with PySCF '
                    '(complex-basis fragments, closed-shell).')
    p.add_argument('xyz', nargs='?', default='system.xyz',
                   help='geometry of the whole system (default: system.xyz)')
    p.add_argument('fragments', nargs='?', default='fragment.txt',
                   help='sobEDA.sh-style fragment definition '
                        '(default: fragment.txt)')
    p.add_argument('--xc', required=True,
                   help='functional, e.g. B3LYP, WB97M-V')
    p.add_argument('--basis', required=True, nargs='+', metavar='SPEC',
                   help="basis set NAME and/or per-element El=NAME, e.g. "
                        "'6-311+G(2d,p)' or '6-311G*'")
    p.add_argument('--ecp', nargs='+', metavar='SPEC',
                   help='ECP NAME and/or per-element El=NAME (default: the '
                        "ECP belonging to each element's basis set, if any)")
    p.add_argument('--disp', choices=DISP_VERSIONS,
                   help='dispersion correction (default: none)')
    p.add_argument('--sobedaw', nargs=3, type=float, metavar=('C', 'A', 'R'),
                   help='also report sobEDAw terms with these parameters')
    p.add_argument('--cart', action='store_true',
                   help='Cartesian d/f functions (Gaussian uses 6D, 7F for '
                        '6-31G*)')
    p.add_argument('--grid-level', type=int, default=GRID_LEVEL,
                   help=f'pyscf DFT grid level (default: {GRID_LEVEL})')
    p.add_argument('--df', choices=['off', 'rijk', 'rij'], default='off',
                   help='density fitting (default: off)')
    p.add_argument('--no-pairwise', action='store_true',
                   help='skip the pairwise / many-body decomposition '
                        '(>2 fragments)')
    p.add_argument('--gpu', action='store_true', help='use gpu4pyscf')
    p.add_argument('--threads', type=int,
                   help='OpenMP threads (default: pyscf)')
    p.add_argument('--outdir', default='output',
                   help='directory for JSON/chk/molden (default: output)')
    p.add_argument('--no-chk', action='store_true',
                   help='do not keep .chk checkpoint files (no restart)')
    p.add_argument('--no-molden', action='store_true',
                   help='do not write .molden files')
    p.add_argument('--cubes', action='store_true',
                   help='also write a density cube file for every stage')
    p.add_argument('--verbose', type=int, default=VERBOSE,
                   help=f'pyscf verbosity (default: {VERBOSE})')
    p.add_argument('--logfile',
                   help='write progress messages to this file and each '
                        "SCF's pyscf log to OUTDIR/*.pyscf_out "
                        '(default: stdout)')
    args = p.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        p.error('--threads must be at least 1')
    try:
        args.basis = _parse_spec(args.basis)
        args.ecp = _parse_spec(args.ecp)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    """Run sobEDA from the command line.

    Keyword Arguments:
        argv {list[str] | None} -- Arguments; None reads sys.argv
            (default: {None})
    """
    args = _parse_args(argv)
    _set_backend(args.gpu)
    if args.threads is not None:
        lib.num_threads(args.threads)

    atoms = read_xyz(args.xyz)
    fragments = read_fragments(args.fragments)
    print(f"Read {len(atoms)} atoms from {args.xyz}, "
          f"{len(fragments)} fragments from {args.fragments}")

    disp = _effective_disp(args.xc, args.disp)
    if disp != args.disp:
        print(f"Note: {args.xc} contains VV10 nonlocal correlation; the "
              f"{args.disp} correction is not applied (double counting).")
    df_modes: dict[str, bool | str] = {'off': False, 'rijk': True,
                                       'rij': 'j-only'}
    use_df = df_modes[args.df]
    disp_tag = '-' + disp.upper() if disp else ''
    lvl = f"{args.xc}{disp_tag}/{_spec_str(args.basis)}"
    df_tag = {False: '', True: ', RI-JK', 'j-only': ', RI-J'}[use_df]
    print(f"\n===== sobEDA {lvl} (counterpoise, complex basis{df_tag}) "
          f"=====")
    terms, checks, pairs = sobeda(
        atoms, fragments, args.xc, args.basis, disp=disp, ecp=args.ecp,
        grid_level=args.grid_level, save_outputs=True, save_cubes=args.cubes,
        outdir=args.outdir, verbose=args.verbose, logfile=args.logfile,
        pairwise=not args.no_pairwise, use_df=use_df, cart=args.cart,
        sobedaw_parm=args.sobedaw, save_chk=not args.no_chk,
        save_molden=not args.no_molden)
    terms_w = sobedaw(terms, *args.sobedaw) if args.sobedaw else None
    print_results(terms, checks, pairs, terms_w)


if __name__ == '__main__':
    main()
