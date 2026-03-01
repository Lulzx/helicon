# magnozzlex.runner

Simulation launch, hardware detection, batch submission, checkpoints, and convergence.

## Hardware Detection

::: magnozzlex.runner.hardware_config.HardwareInfo

::: magnozzlex.runner.hardware_config.detect_hardware

## Launch

::: magnozzlex.runner.launch.RunResult

::: magnozzlex.runner.launch.run_simulation

## Batch Submission

::: magnozzlex.runner.batch.BatchConfig

::: magnozzlex.runner.batch.BatchJob

::: magnozzlex.runner.batch.BatchResult

::: magnozzlex.runner.batch.run_local_batch

::: magnozzlex.runner.batch.submit_batch

::: magnozzlex.runner.batch.generate_slurm_script

::: magnozzlex.runner.batch.generate_pbs_script

## Checkpoints

::: magnozzlex.runner.checkpoints.CheckpointInfo

::: magnozzlex.runner.checkpoints.find_checkpoints

::: magnozzlex.runner.checkpoints.find_latest_checkpoint

::: magnozzlex.runner.checkpoints.cleanup_checkpoints

::: magnozzlex.runner.checkpoints.get_restart_flag

## Grid Convergence

::: magnozzlex.runner.convergence.ConvergenceResult

::: magnozzlex.runner.convergence.run_convergence_study
