"""Unit tests for workspace allocation and memory usage."""

import pytest

from PAOFLOW.transport.workspace.workspace import Workspace


@pytest.mark.unit
def test_workspace_allocate_and_memusage():
    """Allocation should create arrays and report non-zero memory usage."""
    workspace = Workspace()
    workspace.allocate(
        dimL=1,
        dimC=2,
        dimR=1,
        dimx_lead=2,
        nkpts_par=3,
        nrtot_par=4,
        write_lead_sgm=True,
        write_gf=True,
    )

    assert workspace.allocated
    assert workspace.sgm_L is not None
    assert workspace.kgC is not None
    assert workspace.memusage() > 0.0


@pytest.mark.unit
def test_workspace_deallocate_clears_arrays():
    """Deallocation should reset arrays to None and clear allocated flag."""
    workspace = Workspace()
    workspace.allocate(
        dimL=1,
        dimC=1,
        dimR=1,
        dimx_lead=1,
        nkpts_par=1,
        nrtot_par=1,
        write_lead_sgm=False,
        write_gf=False,
    )

    workspace.deallocate()

    assert workspace.tsum is None
    assert not workspace.allocated


@pytest.mark.unit
def test_workspace_allocate_twice_raises():
    """Workspace should not allow double allocation."""
    workspace = Workspace()
    workspace.allocate(
        dimL=1,
        dimC=1,
        dimR=1,
        dimx_lead=1,
        nkpts_par=1,
        nrtot_par=1,
        write_lead_sgm=False,
        write_gf=False,
    )

    with pytest.raises(RuntimeError):
        workspace.allocate(
            dimL=1,
            dimC=1,
            dimR=1,
            dimx_lead=1,
            nkpts_par=1,
            nrtot_par=1,
            write_lead_sgm=False,
            write_gf=False,
        )
