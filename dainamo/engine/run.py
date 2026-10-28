"""Subprocess entry point: ``python -m dainamo.engine.run <job_dir>`` (training, a checkpoint evaluation of a
finished job, making a RefMod, extracting a LoRA from a finetune, or merging LoRAs).

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
        result = (trainer.run_evaluation() if trainer.evaluating
                  else trainer.run_refmod() if trainer.s.get("refmod_enabled")
                  else trainer.run_extract() if trainer.s.get("extract_enabled")
                  else trainer.run_merge() if trainer.s.get("merge_enabled") else trainer.run())
        trainer.write_status(force=True)
        sys.exit(3 if result == "stopped" else 0)
    except StopRequested:
        if trainer:
            trainer.status["phase"] = "stopped"
            made = any(trainer.s.get(k) for k in ("refmod_enabled", "extract_enabled", "merge_enabled"))  # nothing trains
            trainer.log("Stopped" if made else "Stopped before training began")
        sys.exit(3)
    except Exception as e:  # report the failure to the UI, then exit non-zero
        traceback.print_exc()
        if trainer and trainer.evaluating:  # the evaluation failed; the job keeps its training outcome
            trainer.end_evaluation("failed", f"{type(e).__name__}: {e}")
            sys.exit(1)
        status = trainer.status if trainer else {}
        status.update(phase="failed", message=f"{type(e).__name__}: {e}", failed_at=time.time())
        from ..fsutil import replace

        path = os.path.join(job_dir, "status.json")
        try:
            with open(path + ".tmp", "w") as f:
                json.dump(status, f)
            replace(path + ".tmp", path)
        except OSError as err:  # a full disk, likely what it failed of: the server still sees it exit
            print(f"Couldn't record the failure in status.json: {err}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
