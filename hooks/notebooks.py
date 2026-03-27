"""MkDocs hook to symlink notebooks from notebooks/ into docs/ for rendering.

Automatically discovers all .ipynb files under notebooks/ and mirrors the
directory structure into docs/ so mkdocs-jupyter can render them.
"""

import logging
import os
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.notebooks")

NOTEBOOKS_DIR = "notebooks"
DOCS_DIR = "docs"
IGNORE_DIRS = {".ipynb_checkpoints"}

# Top-level directories that notebooks reference via relative paths
# (e.g. ../../static/images/dl/foo.png from notebooks/dl/).
# These get symlinked into docs/ so assets resolve in the built site.
ASSET_DIRS = ["static"]

_created_links = []


def _symlink(src, dst):
  """Create a symlink from dst -> src, recording it for cleanup."""
  dst.parent.mkdir(parents=True, exist_ok=True)
  if dst.is_symlink() or dst.exists():
    dst.unlink()
  rel_src = os.path.relpath(src, dst.parent)
  dst.symlink_to(rel_src)
  _created_links.append(dst)


def on_pre_build(config, **_kwargs):
  """Discover and symlink all notebooks and assets into docs/ before build."""
  root = Path(config["config_file_path"]).parent
  notebooks_root = root / NOTEBOOKS_DIR
  docs_root = root / DOCS_DIR

  # Symlink notebooks
  for src in notebooks_root.rglob("*.ipynb"):
    if any(part in IGNORE_DIRS for part in src.parts):
      continue
    rel = src.relative_to(notebooks_root)
    dst = docs_root / rel
    _symlink(src, dst)
    log.info("Linked %s/%s -> %s/%s", DOCS_DIR, rel, NOTEBOOKS_DIR, rel)

  # Symlink asset directories so relative image paths resolve
  for asset_dir in ASSET_DIRS:
    src = root / asset_dir
    if src.is_dir():
      dst = docs_root / asset_dir
      _symlink(src, dst)
      log.info("Linked %s/%s -> %s", DOCS_DIR, asset_dir, asset_dir)


def on_post_build(config, **_kwargs):
  """Clean up symlinks after build."""
  for link in _created_links:
    if link.is_symlink():
      link.unlink()
      # Remove empty parent dirs
      parent = link.parent
      while parent != Path(config["config_file_path"]).parent / "docs":
        try:
          parent.rmdir()
          parent = parent.parent
        except OSError:
          break
  _created_links.clear()
