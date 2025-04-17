# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import sys

# Benchmarks is a scripts folder. However, we want to reuse and import
# scripts within the benchmarks folder and externally. This is done by adding
# the benchmarks' parent folder to the system path and using fully qualified
# module names like benchmarks.configs.names when importing.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.expanduser(os.path.realpath(__file__))))
)


from benchmarks.configs.load import load_configs
from benchmarks.configs.names import NAMES
from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser
from tbp.monty.frameworks.run_env import setup_env
import pydevd
import socket

setup_env()

from tbp.monty.frameworks.run import main  # noqa: E402

if __name__ == "__main__":
    cmd_args = None
    cmd_parser = create_cmd_parser(experiments=NAMES)
    cmd_args = cmd_parser.parse_args()
    experiments = cmd_args.experiments

    if cmd_args.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    CONFIGS = load_configs(experiments)

    # Inject flag into experiment config(s)
    for config in CONFIGS.values():
        config.setdefault("experiment_args", {})
        config["experiment_args"]["show_sensor_output"] = cmd_args.show_sensor_output

    if cmd_args.show_sensor_output:
        print("✅ show-sensor-output detected: adding sensor output visualization.")

    try:
        # Try to open a socket to the debugger
        sock = socket.create_connection(("172.17.96.1", 5678), timeout=1)
        sock.close()

        pydevd.settrace(
            host="172.17.96.1",
            port=5678,
            stdoutToServer=False,
            stderrToServer=False,
            suspend=False,  # Set to False if you don't want to break immediately
            patch_multiprocessing=True
        )
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        print(f"[DEBUG] Skipping debugger attach: {e}")

    main(all_configs=CONFIGS, experiments=cmd_args.experiments)
