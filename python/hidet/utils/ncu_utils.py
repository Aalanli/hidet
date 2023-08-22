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
{python_executable} {python_script} {module_dir} {args_path}
""".replace('\n', ' ').strip()
_ncu_ui_template = "{ncu_ui_path} {report_path}"


class NsightComputeReport:
    def __init__(self, report_path: str):
        self.report_path: str = report_path

    def visualize(self):
        subprocess.run(_ncu_ui_template.format(ncu_ui_path=_ncu_ui_path, report_path=self.report_path), shell=True)


def _ncu_run_compiled_module(module_dir: str, args_pickled_path: str):
    from hidet.runtime import load_compiled_module
    with open(args_pickled_path, 'rb') as f:
        args = pickle.load(f)
    compiled_module = load_compiled_module(module_dir)
    compiled_module(*args)


def ncu_set_path(ncu_path: str):
    global _ncu_path
    _ncu_path = ncu_path


def ncu_run(compiled_runnable: CompiledModule, *args) -> NsightComputeReport:
    ncu_path = _ncu_path
    module_dir: str = compiled_runnable.module_dir

    # find a report path
    report_path: str = os.path.join(module_dir, 'report.ncu-rep')
    idx = 0
    while os.path.exists(report_path):
        report_path = os.path.join(module_dir, f'report{idx}.ncu-rep')
        idx += 1

    # dump args
    args_path: str = os.path.join(module_dir, 'args.pkl')
    with open(args_path, 'wb') as f:
        pickle.dump(args, f)

    subprocess.run(
        _ncu_template.format(
            ncu_path=ncu_path,
            report_path=report_path,
            python_executable=sys.executable,
            python_script=__file__,
            module_dir=module_dir,
            args_path=args_path,
        ),
        shell=True
    )

    return NsightComputeReport(report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('module_dir', type=str)
    parser.add_argument('args_pickled_path', type=str)
    args = parser.parse_args()
    _ncu_run_compiled_module(args.module_dir, args.args_pickled_path)
