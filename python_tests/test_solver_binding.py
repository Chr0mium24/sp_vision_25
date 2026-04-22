from pathlib import Path

import numpy as np

import sp_vision_bindings as svb


def test_solver_reproject_and_solve_roundtrip():
    config = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    solver = svb.Solver(str(config))

    xyz = np.array([1.2, 0.3, 0.5], dtype=float)
    yaw = 0.4

    points = solver.reproject_armor(xyz, yaw, svb.ArmorType.small, svb.ArmorName.one)
    armor = svb.Armor(3, 0.95, (0, 0, 0, 0), points)

    solver.solve(armor)

    assert np.allclose(np.asarray(armor.xyz_in_world, dtype=float), xyz, atol=5e-2)
    assert abs(((float(armor.ypr_in_world[0]) - yaw + np.pi) % (2 * np.pi)) - np.pi) < 1e-1
    assert solver.debug["valid"] is True
    assert solver.debug["yaw_optimized"] is True
