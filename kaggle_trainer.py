"""
kaggle_trainer.py — Fully automated SO-VITS-SVC training on Kaggle free GPU.

Flow (all automatic after one-time key setup):
  1. Zip training data (dataset/44k, logs, configs, filelists)
  2. Upload as private Kaggle dataset
  3. Push a GPU notebook kernel referencing that dataset
  4. Poll Kaggle every 60 s until complete
  5. Download trained G_*.pth + config.json
  6. Install into models/<name>/  ← app picks it up automatically

Kaggle free tier: P100 GPU, 30 hrs/week, up to 12 hrs per session.
Setup: kaggle.com → Settings → API → Create New Token → kaggle.json
"""

import json, os, pathlib, shutil, tempfile, time, zipfile
from pathlib import Path

# ── Kaggle notebook that runs on their GPU ─────────────────────────────────
_NOTEBOOK = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "cells": [
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["!pip install -q so-vits-svc-fork==4.2.30 torchcodec\n",
                    "print('✅ Dependencies ready')"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["import torch\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')\n",
                    "    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\n",
                    "else:\n",
                    "    raise RuntimeError('No GPU available')"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["import zipfile, glob, os\n",
                    "zips = glob.glob('/kaggle/input/**/*.zip', recursive=True)\n",
                    "if not zips: raise FileNotFoundError('No training zip found in /kaggle/input/')\n",
                    "print(f'📦 Extracting: {zips[0]}')\n",
                    "with zipfile.ZipFile(zips[0]) as z:\n",
                    "    z.extractall('/kaggle/working/')\n",
                    "print('✅ Extracted')"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["import glob as g\n",
                    "feats = g.glob('/kaggle/working/dataset/44k/**/*.data.pt', recursive=True)\n",
                    "cfgs  = g.glob('/kaggle/working/configs/44k/config.json')\n",
                    "print(f'Feature files: {len(feats)}  |  Config: {bool(cfgs)}')\n",
                    "if not feats: raise RuntimeError('No feature files found — zip may be incomplete')\n",
                    "if not cfgs:  raise RuntimeError('No config.json found — zip may be incomplete')"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["import os\n",
                    "os.chdir('/kaggle/working')\n",
                    "print('🚀 Training started — this takes several hours...')\n",
                    "!svc train -c configs/44k/config.json -m logs/44k\n",
                    "print('✅ Training loop finished')"]},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
         "source": ["import glob, os\n",
                    "ckpts = sorted(glob.glob('/kaggle/working/logs/44k/G_*.pth'),\n",
                    "               key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p))) or '0'))\n",
                    "print('Saved checkpoints:', [os.path.basename(c) for c in ckpts])\n",
                    "if ckpts: print(f'Best: {ckpts[-1]}')\n",
                    "else: print('⚠️  No checkpoints found — training may not have completed')"]}
    ]
}


class KaggleTrainer:
    """Fully automated Kaggle GPU training for SO-VITS-SVC models."""

    CREDENTIALS = Path.home() / '.kaggle' / 'kaggle.json'

    def __init__(self, base_dir: Path):
        self.base = Path(base_dir)
        self._api = None

    # ── Setup ──────────────────────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        return self.CREDENTIALS.exists()

    def setup_from_file(self, json_path: str):
        """Install kaggle.json from a user-supplied path."""
        src = Path(json_path)
        if not src.exists():
            raise FileNotFoundError(f'Not found: {src}')
        self.CREDENTIALS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, self.CREDENTIALS)
        self.CREDENTIALS.chmod(0o600)

    def kaggle_username(self) -> str:
        if not self.configured:
            return ''
        with open(self.CREDENTIALS) as f:
            return json.load(f).get('username', '')

    # ── Main entry point ───────────────────────────────────────────────────

    def train(self, model_name: str, progress_cb=None, stop_event=None):
        """
        Run the full pipeline. Returns (success: bool, message: str).
        progress_cb(msg: str, pct: int) called throughout.
        stop_event: threading.Event that cancels polling when set.
        """
        def log(msg, pct=-1):
            if progress_cb:
                progress_cb(msg, pct)

        try:
            api      = self._get_api()
            username = self.kaggle_username()
            slug     = self._slugify(model_name)
            ds_slug  = f'aivocals-{slug}'
            k_slug   = f'aivocals-train-{slug}'

            # 1. Package
            log('📦 Packaging training data...', 5)
            zip_path = self._package(model_name)
            log(f'   Zip: {zip_path.stat().st_size / 1e6:.0f} MB', 8)

            # 2. Upload dataset
            log('⬆️  Uploading to Kaggle (may take a few minutes)...', 10)
            self._upload_dataset(api, username, ds_slug, zip_path, model_name)
            log('✅ Dataset uploaded', 28)

            # 3. Push kernel
            log('🚀 Launching GPU training kernel on Kaggle...', 30)
            self._push_kernel(api, username, k_slug, ds_slug, model_name)
            log('   Kernel submitted — waiting for Kaggle to start it...', 33)

            # 4. Poll
            ok = self._poll(api, username, k_slug, log, stop_event)
            if not ok:
                return False, 'Training cancelled or failed on Kaggle'

            # 5. Download + install
            log('⬇️  Downloading trained model from Kaggle...', 90)
            pth = self._download_and_install(api, username, k_slug, model_name)
            log(f'✅ Model installed → models/{model_name}/{pth.name}', 100)
            return True, str(pth)

        except Exception as exc:
            return False, str(exc)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_api(self):
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended
            self._api = KaggleApiExtended()
            self._api.authenticate()
        return self._api

    @staticmethod
    def _slugify(name: str) -> str:
        return name.lower().replace('_', '-').replace(' ', '-')[:30]

    def _package(self, model_name: str) -> Path:
        """Zip dataset/44k/<model>, logs/44k base checkpoints, configs, filelists."""
        tmp   = Path(tempfile.mkdtemp())
        pkg   = tmp / 'pkg'
        dirs  = [
            pkg / 'dataset' / '44k' / model_name,
            pkg / 'logs'    / '44k',
            pkg / 'configs' / '44k',
            pkg / 'filelists' / '44k',
        ]
        for d in dirs:
            d.mkdir(parents=True)

        src_data = self.base / 'dataset' / '44k' / model_name
        for f in src_data.glob('*.data.pt'):
            shutil.copy(f, dirs[0] / f.name)
        for f in src_data.glob('*.wav'):
            shutil.copy(f, dirs[0] / f.name)

        for fname in ('D_0.pth', 'G_0.pth', 'config.json'):
            src = self.base / 'logs' / '44k' / fname
            if src.exists():
                shutil.copy(src, dirs[1] / fname)

        cfg_src = self.base / 'configs' / '44k' / 'config.json'
        if cfg_src.exists():
            shutil.copy(cfg_src, dirs[2] / 'config.json')

        for fname in ('train.txt', 'val.txt', 'test.txt'):
            src = self.base / 'filelists' / '44k' / fname
            if src.exists():
                shutil.copy(src, dirs[3] / fname)

        zip_out = tmp / f'{model_name}_training.zip'
        with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in pkg.rglob('*'):
                if f.is_file():
                    zf.write(f, f.relative_to(pkg))
        shutil.rmtree(pkg)
        return zip_out

    def _upload_dataset(self, api, username, ds_slug, zip_path: Path, title: str):
        """Create or update a private Kaggle dataset with the training zip."""
        tmp = Path(tempfile.mkdtemp())
        shutil.copy(zip_path, tmp / zip_path.name)
        meta = {
            "title": f"AI Vocals Training — {title}",
            "id": f"{username}/{ds_slug}",
            "licenses": [{"name": "other"}],
            "isPrivate": True
        }
        (tmp / 'dataset-metadata.json').write_text(json.dumps(meta))

        # Try create; if exists, create new version
        try:
            api.dataset_create_new(str(tmp), convert_to_csv=False, dir_mode='zip')
        except Exception:
            api.dataset_create_version(str(tmp), version_notes='Updated training data',
                                       convert_to_csv=False, dir_mode='zip')
        shutil.rmtree(tmp)

    def _push_kernel(self, api, username, k_slug, ds_slug, model_name: str):
        """Push a GPU kernel notebook to Kaggle."""
        tmp = Path(tempfile.mkdtemp())
        nb_path = tmp / 'train.ipynb'
        nb_path.write_text(json.dumps(_NOTEBOOK))
        meta = {
            "id": f"{username}/{k_slug}",
            "title": f"AI Vocals Train {model_name}",
            "code_file": "train.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"{username}/{ds_slug}"],
            "competition_sources": [],
            "kernel_sources": []
        }
        (tmp / 'kernel-metadata.json').write_text(json.dumps(meta))
        api.kernel_push(str(tmp))
        shutil.rmtree(tmp)

    def _poll(self, api, username, k_slug, log, stop_event, interval=60) -> bool:
        """Poll Kaggle kernel status until complete/error/cancelled."""
        start = time.time()
        while True:
            if stop_event and stop_event.is_set():
                return False
            try:
                s = api.kernel_status(username, k_slug)
                status   = getattr(s, 'status',        'unknown')
                run_time = getattr(s, 'totalRunningTime', None)
                elapsed  = f'{run_time}s on Kaggle' if run_time else f'{int(time.time()-start)}s local'
                pct = min(35 + int((time.time() - start) / 36), 88)  # creep 35→88 over ~18 hrs
                log(f'   Kaggle status: {status}  ({elapsed})', pct)
                if status == 'complete':
                    return True
                if status in ('error', 'cancel_acknowledged', 'cancel_requested'):
                    log(f'❌ Kaggle kernel ended with status: {status}', -1)
                    return False
            except Exception as e:
                log(f'   Poll error (retrying): {e}', -1)
            time.sleep(interval)

    def _download_and_install(self, api, username, k_slug, model_name: str) -> Path:
        """Download kernel output and install best checkpoint into models/."""
        tmp = Path(tempfile.mkdtemp())
        api.kernel_output(username, k_slug, path=str(tmp))

        # Find best checkpoint in downloaded output
        checkpoints = sorted(
            tmp.rglob('G_*.pth'),
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or '0')
        )
        if not checkpoints:
            raise RuntimeError('No trained checkpoint found in Kaggle output')

        best = checkpoints[-1]
        dest = self.base / 'models' / model_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, dest / best.name)

        cfg_src = next(tmp.rglob('config.json'), None)
        if cfg_src:
            shutil.copy(cfg_src, dest / 'config.json')

        shutil.rmtree(tmp)
        return dest / best.name
