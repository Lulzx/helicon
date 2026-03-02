# helicon.runner

Simulation launch, hardware detection, batch submission, checkpoints, and convergence.

## Hardware Detection

::: helicon.runner.hardware_config.HardwareInfo

::: helicon.runner.hardware_config.detect_hardware

## Launch

::: helicon.runner.launch.RunResult

::: helicon.runner.launch.run_simulation

## Batch Submission

::: helicon.runner.batch.BatchConfig

::: helicon.runner.batch.BatchJob

::: helicon.runner.batch.BatchResult

::: helicon.runner.batch.run_local_batch

::: helicon.runner.batch.submit_batch

::: helicon.runner.batch.generate_slurm_script

::: helicon.runner.batch.generate_pbs_script

## Checkpoints

::: helicon.runner.checkpoints.CheckpointInfo

::: helicon.runner.checkpoints.find_checkpoints

::: helicon.runner.checkpoints.find_latest_checkpoint

::: helicon.runner.checkpoints.cleanup_checkpoints

::: helicon.runner.checkpoints.get_restart_flag

## Grid Convergence

::: helicon.runner.convergence.ConvergenceResult

::: helicon.runner.convergence.run_convergence_study
