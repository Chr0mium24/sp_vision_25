from pathlib import Path

import sp_vision_bindings as svb


def test_aimer_aim_returns_command_tuple_and_debug():
    config = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    solver = svb.Solver(str(config))
    target = svb.Target.from_armor(
        svb.Armor(4, 0.95, (0, 0, 0, 0), solver.reproject_armor(
            [1.1, 0.2, 0.3], 0.15, svb.ArmorType.small, svb.ArmorName.one
        )),
        radius=0.2,
        armor_num=4,
    )

    aimer = svb.Aimer(str(config))
    command = aimer.aim([target], 25.0)

    assert isinstance(command, tuple)
    assert len(command) == 5
    assert isinstance(aimer.debug, dict)
    assert "valid" in aimer.debug
