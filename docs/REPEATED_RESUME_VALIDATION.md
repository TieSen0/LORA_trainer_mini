# Real GPU repeated-resume validation

This is a manual test plan, not a test result. Preparing this document does not
launch training. Automated CPU fixtures exercise the app's resume path; this
check is for the actual GPU trainer and its serialized state.

## Before starting

- Run only when you are ready to use the GPU and allocate space for several
  checkpoints. Startup/model loading repeats for every continuation.
- Do not interrupt an important training job. Close other GPU workloads first.
- Copy a small set of images and matching captions into a **new validation
  dataset folder**. Do not change the production dataset for this experiment.
- Use a separate output folder, such as `output/resume-validation/`. Keep every
  resulting run folder until verification is finished. State files can be much
  larger than the final LoRA; the app does not automatically delete them.
- Use the same trainer, base model, dataset, and training settings throughout.
  Only load checkpoints you created or trust. Do not edit state files manually.

## Suggested small setup

Start with your known-working SDXL setup and save a separate settings copy.
Select the copied dataset and new output folder. Use four readable image/caption
pairs, batch size 1, repeats 1, three total epochs, a fixed seed, resumable state
saving enabled, and checkpoint interval 2. Keep your known-working precision,
attention, optimizer, resolution, and memory settings.

The trainer determines the real step count after dataset preparation. With four
batches per epoch, the example target is 12 steps. Verify the logged batches per
epoch and total. If they differ, adapt the checkpoints below so you test both
mid-epoch and epoch-boundary continuation. Startup and caching may take longer
than the small number of training steps; no runtime estimate is guaranteed.

## Test sequence

| Run | Starting point | Action | Expected evidence |
| --- | --- | --- | --- |
| A | Fresh training | Wait for a confirmed checkpoint at step 2, then Stop and confirm. | Checkpoint folder exists; required files passed checks; reported unsaved-step count is plausible. |
| B | Run A's step-2 checkpoint | Prepare Resume with the original 3-epoch target. Save a later checkpoint, such as step 4, then Stop. | State-load messages; first new overall step is 3; new output folder. |
| C | Run B's step-4 checkpoint | Resume again, save a checkpoint such as step 6, then Stop. | State-load messages; first new overall step is 5; continuation crosses an epoch boundary correctly. |
| D | Run C's step-6 checkpoint | Resume again and let the remaining work finish. | First new overall step is 7; target remains 12; successful exit and final model file. |

Training can advance while the Stop question is open. Stop does not create a new
checkpoint. If more checkpoints were saved than expected, select the intended
earlier one in History or record the later step and adapt the next row. Do not
claim Stop occurred at exactly the step displayed when you clicked the button.

Select the run and state in **History / Compare → Prepare Resume**, or use
**Prepare Latest Resume** in the Training Log tab after stopping. Settings are
filled in, but nothing starts automatically. Review, press Resume Training, and
approve the trusted-state prompt. Do not clear the resume field or change Epochs
to the number of remaining epochs.

## What to inspect

1. **Restored state:** logs report successful model, optimizer, scheduler, random
   state, and (for fp16) gradient-scaler restoration. The confirmation dialog
   alone is not proof that restoration succeeded.
2. **Position:** the first new app-reported step is the chosen overall saved step
   plus 1. Raw trainer progress and saved `current_step` may be epoch-relative;
   the app combines these with the saved epoch and steps-per-epoch metadata.
3. **Checkpoint status:** the last complete checkpoint advances only after save
   completion evidence and required-file checks. A folder appearing by itself
   is not enough. Open Checkpoint Folder opens the displayed state directory.
4. **Work at risk:** the displayed count is reported overall progress minus the
   recovery-point step, never negative. A resumed run with no new checkpoint
   still shows its selected source checkpoint. This is an estimate: training can
   advance before the UI receives output or while Stop confirmation is open.
5. **Preservation:** each continuation has a distinct run directory. Source state
   files are unchanged. Select checkpoints from the immediately previous resumed
   run rather than repeatedly selecting Run A's checkpoint.
6. **Completion:** the total target remains unchanged; an extra full schedule is
   not started. The final model is nonempty and loads in your SDXL inference
   workflow. Completion alone does not establish image quality.

If restoration reports errors, position is wrong, or checkpoint status disagrees
with files/logs, stop and retain the run folders for diagnosis. Do not guess at
metadata repairs or continue before the discrepancy is understood.

## Record the result

Record these fields in your validation notes; remove private paths before sharing:

- App Git commit, sd-scripts revision, Python/PyTorch/Accelerate versions.
- GPU, precision, optimizer, scheduler, attention backend, and base model identity.
- Dataset image count, batches per epoch, original epoch/step target, and seed.
- Per run: selected source, saved overall step, first new step, latest complete
  checkpoint, reported steps at risk, exit status, and relevant log lines.
- Whether every restoration component and the final inference load succeeded.

**Pass:** all four runs follow the intended chain, restoration succeeds, progress
and completion are correct, and source checkpoints remain intact. This verifies
continuation behavior, not bit-for-bit equivalence to uninterrupted training or
improved model quality. A controlled uninterrupted baseline is a separate
experiment if numerical reproducibility is required.

**Current status: not run.** No GPU experiment was started while preparing this plan.
