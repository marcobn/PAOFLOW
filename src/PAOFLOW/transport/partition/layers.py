from __future__ import annotations

from collections.abc import Iterable

from PAOFLOW.DataController import DataController
from PAOFLOW.transport.partition.directions import direction_index, normalize_transport_direction
from PAOFLOW.transport.partition.orbital_inventory import build_atom_orbital_inventory
from PAOFLOW.transport.partition.types import AtomOrbitals, HamiltonianBlockPartition


def resolve_layer_partition(
    data_controller: DataController,
    *,
    central_atoms: str = 'ALL',
    central_layers: int | None = None,
    left_lead_layers: int | None = None,
    right_lead_layers: int | None = None,
    transport_direction: str,
    layer_tolerance: float = 1.0e-6,
) -> HamiltonianBlockPartition:
    direction = normalize_transport_direction(transport_direction)
    inventory = build_atom_orbital_inventory(data_controller)
    layers = _group_atoms_into_layers(inventory, direction, layer_tolerance)

    if central_layers is not None:
        _validate_layer_count('central_layers', central_layers, len(layers))
        central_orbitals = _layers_to_orbitals(layers[:central_layers])
    elif central_atoms.strip().lower() == 'all':
        central_orbitals = inventory
    else:
        raise ValueError(
            "central_atoms currently supports only 'ALL'; use central_layers for layers."
        )

    dim_c = _dimension(central_orbitals)
    selectors: dict[str, dict[str, str]] = {
        'H00_C': {'rows': _range_string(central_orbitals), 'cols': _range_string(central_orbitals)},
    }

    if left_lead_layers is None and right_lead_layers is None:
        coupling_orbitals = (
            _layers_to_orbitals(layers[-central_layers:]) if central_layers else central_orbitals
        )
        selectors['H_CR'] = {
            'rows': _range_string(coupling_orbitals),
            'cols': _range_string(central_orbitals),
        }
        return HamiltonianBlockPartition(
            dim_c=dim_c,
            dim_l=None,
            dim_r=None,
            transport_direction=direction,
            selectors=selectors,
        )

    if left_lead_layers is None or right_lead_layers is None:
        raise ValueError('left_lead_layers and right_lead_layers must be provided together.')
    _validate_layer_count('left_lead_layers', left_lead_layers, len(layers))
    _validate_layer_count('right_lead_layers', right_lead_layers, len(layers))

    left_orbitals = _layers_to_orbitals(layers[:left_lead_layers])
    right_orbitals = _layers_to_orbitals(layers[-right_lead_layers:])
    dim_l = _dimension(left_orbitals)
    right_dimension = _dimension(right_orbitals)
    if right_dimension != dim_l:
        raise ValueError(
            'Left and right lead layers must contain the same number of PAO orbitals: '
            f'{dim_l} != {right_dimension}.'
        )
    dim_r = right_dimension

    selectors.update(
        {
            'H_CR': {'rows': _range_string(central_orbitals), 'cols': _range_string(left_orbitals)},
            'H_LC': {
                'rows': _range_string(right_orbitals),
                'cols': _range_string(central_orbitals),
            },
            'H00_L': {'rows': _range_string(left_orbitals), 'cols': _range_string(left_orbitals)},
            'H01_L': {'rows': _range_string(right_orbitals), 'cols': _range_string(left_orbitals)},
            'H00_R': {'rows': _range_string(left_orbitals), 'cols': _range_string(left_orbitals)},
            'H01_R': {'rows': _range_string(right_orbitals), 'cols': _range_string(left_orbitals)},
        }
    )
    return HamiltonianBlockPartition(
        dim_c=dim_c,
        dim_l=dim_l,
        dim_r=dim_r,
        transport_direction=direction,
        selectors=selectors,
    )


def _group_atoms_into_layers(
    inventory: list[AtomOrbitals], direction: str, tolerance: float
) -> list[list[AtomOrbitals]]:
    axis = direction_index(direction)
    sorted_atoms = sorted(inventory, key=lambda atom: atom.position[axis])
    layers: list[list[AtomOrbitals]] = []
    layer_coordinate: float | None = None
    for atom in sorted_atoms:
        coordinate = float(atom.position[axis])
        if layer_coordinate is None or abs(coordinate - layer_coordinate) > tolerance:
            layers.append([atom])
            layer_coordinate = coordinate
        else:
            layers[-1].append(atom)
    return layers


def _layers_to_orbitals(layers: Iterable[Iterable[AtomOrbitals]]) -> list[AtomOrbitals]:
    return [atom for layer in layers for atom in layer]


def _validate_layer_count(name: str, value: int, available_layers: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    if value > available_layers:
        raise ValueError(f'{name}={value} exceeds the {available_layers} available layers.')


def _dimension(atoms: list[AtomOrbitals]) -> int:
    return sum(atom.dimension for atom in atoms)


def _range_string(atoms: list[AtomOrbitals]) -> str:
    sorted_atoms = sorted(atoms, key=lambda item: item.first_orbital)
    if not sorted_atoms:
        raise ValueError('Cannot build an orbital range from an empty atom selection.')

    merged_ranges: list[tuple[int, int]] = []
    start = sorted_atoms[0].first_orbital
    end = sorted_atoms[0].last_orbital
    for atom in sorted_atoms[1:]:
        if atom.first_orbital == end + 1:
            end = atom.last_orbital
        else:
            merged_ranges.append((start, end))
            start = atom.first_orbital
            end = atom.last_orbital
    merged_ranges.append((start, end))

    return ','.join(
        str(first) if first == last else f'{first}-{last}' for first, last in merged_ranges
    )
