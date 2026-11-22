
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix


def main():
    # Locate the workspace from the installed ROS 2 package.
    package_prefix = Path(get_package_prefix("tissue_robot"))
    workspace = package_prefix.parent.parent

    # Locate the existing perception system.
    vision_directory = workspace / "src" / "vision_system"
    monitor_script = vision_directory / "sneeze_monitor.py"

    # Use the Python environment containing TensorFlow and MediaPipe.
    python_executable = vision_directory / ".venv" / "bin" / "python"

    if not monitor_script.is_file():
        raise FileNotFoundError(
            f"Could not find perception script: {monitor_script}"
        )

    if not python_executable.is_file():
        raise FileNotFoundError(
            f"Could not find virtual environment Python: {python_executable}"
        )

    # Preserve the relative paths used by the MediaPipe model files.
    os.chdir(vision_directory)

    print("[ACHOO BOT] Starting perception system...")

    # Replace this launcher with the existing perception script.
    os.execv(
        str(python_executable),
        [str(python_executable), str(monitor_script)]
    )


if __name__ == "__main__":
    main()
