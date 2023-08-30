from typing import Optional
import os
import pickle
import argparse
import sys
import subprocess
from hidet.runtime import CompiledModule

_ncu_path: str = '/usr/local/cuda/bin/ncu'
_ncu_ui_path: str = '/usr/local/cuda/bin/ncu-ui'
_ncu_template = """
{ncu_path}
--export {report_path}
--force-overwrite
--target-processes application-only
--replay-mode kernel
--kernel-name-base function
--launch-skip-before-match 0
--filter-mode global
--section ComputeWorkloadAnalysis
--section InstructionStats
--section LaunchStats
--section MemoryWorkloadAnalysis
--section Occupancy
--section SchedulerStats
--section SourceCounters
--section SpeedOfLight
--section SpeedOfLight_RooflineChart
--section WarpStateStats
--sampling-interval auto
--sampling-max-passes 5
--sampling-buffer-size 33554432
--profile-from-start 1
--cache-control all
--clock-control base
--rule CPIStall
--rule FPInstructions
--rule HighPipeUtilization
--rule IssueSlotUtilization
--rule LaunchConfiguration
--rule Occupancy
--rule PCSamplingData
--rule SOLBottleneck
--rule SOLFPRoofline
--rule SlowPipeLimiter
--rule ThreadDivergence
--rule UncoalescedGlobalAccess
--rule UncoalescedSharedAccess
--import-source yes
--check-exit-code yes
{python_executable} {python_script} {args}
""".replace('\n', ' ').strip()
_ncu_ui_template = "{ncu_ui_path} {report_path}"


class NsightComputeReport:
    def __init__(self, report_path: str):
        self.report_path: str = report_path

    def visualize(self):
        subprocess.run(_ncu_ui_template.format(ncu_ui_path=_ncu_ui_path, report_path=self.report_path), shell=True)


def _ncu_run_func(script_path, func_name, args_pickled_path):
    with open(args_pickled_path, 'rb') as f:
        args = pickle.load(f)

    try:
        sys.path.append(os.path.dirname(script_path))
        module = __import__(os.path.basename(script_path)[:-3])
    except Exception as e:
        raise RuntimeError('Can not import the python script: {}'.format(script_path)) from e

    if not hasattr(module, func_name):
        raise RuntimeError('Can not find the function "{}" in {}'.format(func_name, script_path))

    func = getattr(module, func_name)

    try:
        func(*args)
    except Exception as e:
        raise RuntimeError('Error when running the function "{}"'.format(func_name)) from e


def ncu_set_path(ncu_path: str):
    global _ncu_path
    _ncu_path = ncu_path


def ncu_run(func, *args) -> NsightComputeReport:
    import inspect
    import tempfile

    # get the python script path and function name
    script_path: str = inspect.getfile(func)
    func_name: str = func.__name__

    # report path
    report_path: str = os.path.join(os.path.dirname(script_path), 'report.ncu-rep')

    # dump args
    args_path: str = tempfile.mktemp() + '.pkl'
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)

    subprocess.run(
        _ncu_template.format(
            ncu_path=_ncu_path,
            report_path=report_path,
            python_executable=sys.executable,
            python_script=__file__,
            args='{} {} {}'.format(script_path, func_name, args_path)
        ),
        shell=True
    )

    return NsightComputeReport(report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script_path', type=str)
    parser.add_argument('func', type=str)
    parser.add_argument('args', type=str)
    args = parser.parse_args()
    _ncu_run_func(args.script_path, args.func, args.args)
