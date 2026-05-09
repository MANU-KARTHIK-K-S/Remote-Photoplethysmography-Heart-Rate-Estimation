##############################################################################
#  Makefile — HR Estimation from NIR Facial Video (MR-NIRP-D)
##############################################################################
#
#  Prerequisites:
#    pip install -r requirements.txt
#    Set GCP project in configs/config.yaml (gcs.project_id, gcs.bucket_name)
#
#  Typical workflow:
#    1. make upload       # upload PGM dataset to GCS (run once)
#    2. make preprocess   # download + preprocess → HDF5 shards
#    3. make train        # train PhysFormer-NIR
#    4. make evaluate     # evaluate best checkpoint on test set
##############################################################################

.PHONY: all install upload preprocess inspect-mat train evaluate clean help

PYTHON      ?= python3
CONFIG      ?= configs/config.yaml
DATASET_ROOT?= /tmp/mr_nirp_data     # local path to downloaded MR-NIRP-D
CHECKPOINT  ?= checkpoints/physformer_nir_best.pth
EVAL_SPLIT  ?= test

##─── Installation ───────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install -r requirements.txt

##─── Step 1: Upload raw PGMs to GCS (run once after Google Drive download) ──

upload:
	@echo "► Uploading MR-NIRP-D PGM files to GCS..."
	$(PYTHON) scripts/upload_to_gcs.py \
	    --config $(CONFIG) \
	    --dataset-root $(DATASET_ROOT)

upload-dry:
	$(PYTHON) scripts/upload_to_gcs.py \
	    --config $(CONFIG) \
	    --dataset-root $(DATASET_ROOT) \
	    --dry-run

##─── Step 2: Preprocess (PGM → face crop → HDF5 clips) ─────────────────────
#
#  Dark-scene handling:
#    - Session-level linear stretch preserves inter-frame BVP signal
#    - Dead sessions (mean < 1%): stretch + noise injection
#    - All clips kept; dark clips get loss_weight=0.3 in metadata
#
#  GT derivation:
#    - pulseOx.mat (125 Hz BVP) + cam0_full_log.txt (Unix timestamps)
#    - Bandpass [0.75, 3.0] Hz → interpolate to camera fps → z-score

preprocess:
	@echo "► Preprocessing MR-NIRP-D (PGM → HDF5) ..."
	$(PYTHON) scripts/preprocess_dataset.py \
	    --config $(CONFIG) \
	    --dataset-root $(DATASET_ROOT) \
	    --no-upload

preprocess-from-gcs:
	@echo "► Preprocessing from GCS ..."
	$(PYTHON) scripts/preprocess_dataset.py \
	    --config $(CONFIG) \
	    --from-gcs

# Inspect a pulseOx.mat file (diagnostic)
inspect-mat:
	$(PYTHON) scripts/inspect_mat.py \
	    --dataset-root $(DATASET_ROOT)

# Inspect specific session
inspect-session:
	$(PYTHON) scripts/inspect_mat.py \
	    --mat $(DATASET_ROOT)/Subject1/PulseOX/pulseOx.mat \
	    --log $(DATASET_ROOT)/Subject1/resting_indoor/cam0_full_log.txt \
	    --plot

##─── Step 3: Train ──────────────────────────────────────────────────────────

train:
	@echo "► Training PhysFormer-NIR ..."
	$(PYTHON) scripts/train.py --config $(CONFIG)

# Resume from last checkpoint
train-resume:
	$(PYTHON) scripts/train.py \
	    --config $(CONFIG) \
	    --resume checkpoints/physformer_nir_last.pth

# Quick sanity run (2 epochs)
train-sanity:
	$(PYTHON) scripts/train.py \
	    --config $(CONFIG) \
	    training.epochs=2 \
	    training.val_every_n_epochs=1 \
	    training.batch_size=2

##─── Step 4: Evaluate ───────────────────────────────────────────────────────

evaluate:
	@echo "► Evaluating on $(EVAL_SPLIT) split ..."
	$(PYTHON) scripts/evaluate.py \
	    --config $(CONFIG) \
	    --checkpoint $(CHECKPOINT) \
	    --split $(EVAL_SPLIT) \
	    --save-dir eval_outputs

evaluate-val:
	$(MAKE) evaluate EVAL_SPLIT=val

##─── Utilities ──────────────────────────────────────────────────────────────

# Run all import checks (no GPU needed)
check:
	@echo "► Checking imports ..."
	$(PYTHON) -c "from src.data.pgm_loader import PGMSessionLoader; print('pgm_loader OK')"
	$(PYTHON) -c "from src.data.gt_loader import load_session_gt; print('gt_loader OK')"
	$(PYTHON) -c "from src.data.preprocessing import process_session; print('preprocessing OK')"
	$(PYTHON) -c "from src.data.dataset import MRNIRPDataset; print('dataset OK')"
	$(PYTHON) -c "from src.models.physformer_nir import build_model; print('model OK')"
	$(PYTHON) -c "from src.losses.losses import build_loss; print('losses OK')"
	$(PYTHON) -c "from src.training.trainer import Trainer; print('trainer OK')"
	$(PYTHON) -c "from src.evaluation.metrics import HRMetrics; print('metrics OK')"
	@echo "All imports OK ✓"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

help:
	@echo ""
	@echo "HR Estimation from NIR Facial Video (MR-NIRP-D)"
	@echo "================================================="
	@echo ""
	@echo "  make install              Install Python dependencies"
	@echo "  make upload               Upload PGM dataset to GCS"
	@echo "  make preprocess           Preprocess: PGM → HDF5 clips"
	@echo "  make inspect-mat          Inspect pulseOx.mat files"
	@echo "  make train                Train PhysFormer-NIR"
	@echo "  make train-resume         Resume training from last checkpoint"
	@echo "  make train-sanity         Quick 2-epoch sanity check"
	@echo "  make evaluate             Evaluate best checkpoint (test set)"
	@echo "  make check                Verify all imports work"
	@echo ""
	@echo "  Overrides:"
	@echo "    DATASET_ROOT=/path/to/mr-nirp-d  (local dataset path)"
	@echo "    CONFIG=configs/config.yaml"
	@echo "    CHECKPOINT=checkpoints/physformer_nir_best.pth"
	@echo ""

##─── Real GCS workflow (for T4 VM) ──────────────────────────────────────────

# Step 1: verify you can see the subjects
list-subjects:
	gcloud storage ls gs://docs-ingest-bucket/ | grep subject

# Step 2: inspect one session's GT before committing to full preprocess
inspect-one:
	$(PYTHON) scripts/inspect_mat.py \
	    --mat /tmp/mr_nirp_cache/subject14_garage_still_975/PulseOX_subject14_garage_still_975/PulseOX/pulseOx.mat \
	    --log /tmp/mr_nirp_cache/subject14_garage_still_975/PulseOX_subject14_garage_still_975/PulseOX/cam0_full_log.txt

# Step 3: preprocess one subject only (sanity test)
preprocess-one:
	$(PYTHON) scripts/preprocess_dataset.py \
	    --config $(CONFIG) \
	    --subjects 14 \
	    --no-upload

# Step 4: full preprocess + upload
preprocess-gcs:
	$(PYTHON) scripts/preprocess_dataset.py \
	    --config $(CONFIG)
