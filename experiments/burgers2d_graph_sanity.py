from pathlib import Path
import sys

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from workloads.burgers2d import (
    Burgers2DWorkload,
)


def main():
    w = Burgers2DWorkload(
        nx=8,
        ny=8,
        timesteps=100,
        dt=0.004,
        viscosity=0.002,
        advection_scale=1.0,
    )

    checkpoints = [
        1,
        5,
        10,
        20,
        40,
        60,
        80,
        100,
    ]

    print(
        "\n=== 2-D BURGERS GRAPH SANITY ==="
    )

    for t in checkpoints:
        frac = (
            w.nonlocal_edge_fraction(
                t
            )
        )

        print(
            f"t={t:3d}  "
            f"nonlocal_edge_fraction="
            f"{frac:.4f}"
        )

    late = (
        w.nonlocal_edge_fraction(
            100
        )
    )

    if late < 0.25:
        raise AssertionError(
            "Workload still generates too few nonlocal dependencies. "
            "Do not run the expensive NoC experiment yet."
        )

    print(
        "\nGraph sanity passed: "
        "the workload develops meaningful nonlocal communication."
    )


if __name__ == "__main__":
    main()
