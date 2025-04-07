"""Module containing I/O utilities for InstaMap.
"""

from __future__ import annotations

import logging
import gzip
import os

from typing import Dict, TextIO, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def open_with_compression(filename: 'StrOrBytesPath', mode='rt') -> TextIO:
    """Opens a file, automatically decompressing it if it is compressed.

    The compression is determined by the file extension.
    """
    if os.path.splitext(filename)[1] == '.gz':
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


def read_pdb_file(filename: 'StrOrBytesPath',
                  dtype: torch.dtype = torch.float32,
                  zero_center_of_mass: bool = False,
                  ) -> Dict[int, torch.Tensor]:
    """Read a PDB file.

    This function uses a custom implementation of PDB file reading to handle pdb
    files with missing atom types. When the atom type is missing, it is assumed
    to be a hydrogen atom. This ensures compatibility with a wider range of MD
    software.

    Parameters
    ----------
    filename : StrOrBytesPath
        Path to the PDB file to read.
        It may include gzip compressed paths, which will be automatically decompressed.
    dtype : torch.dtype, optional
        The dtype of the returned tensors, by default torch.float32

    Returns
    -------
    Dict[int, torch.Tensor]
        A dictionary mapping atomic numbers to tensors of shape (N, 3) containing
        the positions of the atoms with that atomic number.
    """
    import ase.atoms
    from ase.io import proteindatabank as pdb_io

    logger = logging.getLogger(__name__)

    with open_with_compression(filename) as f:
        lines = f.readlines()

    symbols = []
    coords = []

    for i, l in enumerate(lines):
        if not l.startswith('ATOM'):
            continue

        try:
            l_data = pdb_io.read_atom_line(l)
        except:
            logger.exception('Failed to parse line %d: %s', i, l)
            raise

        symbol = l_data[0]
        name = l_data[1]
        coord = l_data[4]

        if symbol == '':
            symbol = name[0]

        symbols.append(symbol)
        coords.append(coord)

    atoms = ase.atoms.Atoms(symbols, positions=np.stack(coords, axis=0))

    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    if zero_center_of_mass: positions -= atoms.get_center_of_mass()

    return {
        z: torch.from_numpy(positions[atomic_numbers == z]).to(dtype=dtype)
        for z in np.unique(atomic_numbers)
    }


def write_pdb_file(data: Dict[int, torch.Tensor], filename: 'StrOrBytesPath'):
    """Write a given particle to a PDB file.

    Parameters
    ----------
    data : Dict[int, torch.Tensor]
        A dictionary mapping atomic numbers to tensors of shape (N, 3) containing
        the positions of the atoms with that atomic number.
    filename : StrOrBytesPath
        Path to the PDB file to write.
    """
    import ase.atoms
    import ase.io.proteindatabank

    atom_numbers = []
    atom_positions = []

    for z, pos in data.items():
        atom_numbers.append(torch.full((pos.shape[0],), z, dtype=torch.int64))
        atom_positions.append(pos)

    atoms = ase.atoms.Atoms(
        numbers=torch.cat(atom_numbers, dim=0).numpy(),
        positions=torch.cat(atom_positions, dim=0).numpy())

    with open_with_compression(filename, 'wt') as f:
        ase.io.proteindatabank.write_proteindatabank(f, atoms)


def read_gro_file(
    filename: 'StrOrBytesPath',
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Read a gromacs file.

    Custom line-by-line parsing of gromacs files.
    This extracts only the main atom type and position information.
    """
    import ase.atoms

    with open_with_compression(filename) as f:
        lines = iter(f)
        next(lines)
        num_atoms = int(next(lines))

        positions = np.zeros((num_atoms, 3), dtype=np.float32)
        symbols = []

        for i in range(num_atoms):
            l = next(lines)

            symbols.append(l[10:15].strip()[0])
            positions[i, 0] = float(l[20:28]) * 10
            positions[i, 1] = float(l[28:36]) * 10
            positions[i, 2] = float(l[36:44]) * 10

        box_vector_line = next(lines).split()
        box = torch.tensor([float(x) * 10 for x in box_vector_line[:3]], dtype=dtype)

    atoms = ase.atoms.Atoms(symbols, positions=positions)
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()

    return {
        z: torch.from_numpy(positions[atomic_numbers == z]).to(dtype)
        for z in np.unique(atomic_numbers)
    }, box

