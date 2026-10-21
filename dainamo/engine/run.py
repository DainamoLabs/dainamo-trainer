"""Subprocess entry point: ``python -m dainamo.engine.run <job_dir>`` (training, or a checkpoint
evaluation of a finished job).

Exit codes: 0 completed, 3 stopped by user, 1 failed.
"""

import json
import os
import signal
import sys
import time
import traceback


def main():
    job_dir = sys.argv[1]
    from diffusers.utils import logging as diffusers_logging

    from .trainer import StopRequested, Trainer, request_stop

    # diffusers warns about fp32 module lists on every VAE .to(); keep job logs readable
    diffusers_logging.set_verbosity_error()

    # stop gracefully (save state) when the server or the OS asks us to terminate
    signal.signal(signal.SIGTERM, lambda *_: request_stop())

    trainer = None
    try:
        trainer = Trainer(job_dir)
        result = trainer.run_evaluation() if trainer.evaluating else trainer.run()
        trainer.write_status(force=True)
        sys.exit(3 if result == "stopped" else 0)
    except StopRequested:
        if trainer:
            trainer.status["phase"] = "stopped"
            trainer.log("Stopped before training began")
        sys.exit(3)
    except Exception as e:  # report the failure to the UI, then exit non-zero
        traceback.print_exc()
        if trainer and trainer.evaluating:  # the evaluation failed; the job keeps its training outcome
            trainer.end_evaluation("failed", f"{type(e).__name__}: {e}")
            sys.exit(1)
        status = trainer.status if trainer else {}
        status.update(phase="failed", message=f"{type(e).__name__}: {e}", failed_at=time.time())
        with open(os.path.join(job_dir, "status.json"), "w") as f:
            json.dump(status, f)
        sys.exit(1)


if __name__ == "__main__":
    main()
