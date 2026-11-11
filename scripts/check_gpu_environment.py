import os
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


def check_nvidia_gpu() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            text=True,
        ).strip()
        return {"available": True, "details": out}
    except Exception as exc:
        return {"available": False, "details": str(exc)}


def check_mujoco_rendering() -> dict:
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 1"/>
        <geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.1 0.1 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=224, width=224)
    renderer.update_scene(data)
    img = renderer.render()
    return {
        "shape": tuple(img.shape),
        "dtype": str(img.dtype),
        "non_empty": bool(np.any(img > 0)),
    }


def main() -> None:
    print(f"Platform: {platform.platform()}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"MuJoCo:   {mujoco.__version__}")

    gpu = check_nvidia_gpu()
    print(f"GPU:      {gpu['details']}")

    render_info = check_mujoco_rendering()
    print(f"Renderer: shape={render_info['shape']}, dtype={render_info['dtype']}, non_empty={render_info['non_empty']}")


if __name__ == "__main__":
    main()
