from pathlib import Path

import numpy as np

import sp_vision_bindings as svb


def test_target_from_armor_predict_update():
    config = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    solver = svb.Solver(str(config))

    xyz = np.array([1.4, -0.2, 0.35], dtype=float)
    yaw = -0.25
    points = solver.reproject_armor(xyz, yaw, svb.ArmorType.small, svb.ArmorName.two)
    armor = svb.Armor(4, 0.92, (0, 0, 0, 0), points)
    solver.solve(armor)

    target = svb.Target.from_armor(armor, radius=0.2, armor_num=4)

    assert target.isinit is False or isinstance(target.isinit, bool)
    assert len(target.armor_xyza_list) == 4

    target.predict(0.01)
    target.update(armor)

    assert target.debug["last_predict"]["valid"] is True
    assert target.debug["last_update"]["valid"] is True
    assert target.debug["last_update"]["candidate_count"] >= 1
