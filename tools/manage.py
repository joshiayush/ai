#!/usr/bin/python3

# Copyright 2018 The AI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Repository management tool for AI."""

from __future__ import annotations

import os
import sys
import pathlib
import argparse

from datetime import date

import ai_logs
import ai_docs
import ai_utils

_BASE_AI_DIR = 'notebooks'
_BASE_DOCS_DIR = 'docs'
_BASE_AI_REPO = pathlib.Path(__file__).parent.parent


def GenerateAIDocs() -> int:
  """Completes the action of command "--generate-docs" by generating markdown
  files from the IPython Notebooks.

  Returns:
    Error code of `PermissionError`.
  """
  error_code = 0
  base_docs_dir = pathlib.Path(os.getcwd()) / _BASE_DOCS_DIR
  for file in ai_utils.GetFileByExtensionUnderDirectory(
    'ipynb', os.fspath(pathlib.Path(os.getcwd()) / _BASE_AI_DIR)
  ):
    try:
      ai_docs.GenerateDocs(
        base_docs_dir, file, ai_docs.ReadIPythonNotebookToMarkdown(file)
      )
    except PermissionError as exc:
      error_code = exc.errno
  return error_code


_BASE_AI_MODULE_PATH = {'ai': ('ai/', )}
_BASE_AI_DOCS_PATH = {
  'docs': (
    'docs/',
    'notebooks/',
    'templates/',
  )
}


def GenerateAILogs() -> int:
  """Completes the action of command "--generate-logs" by generating changelog
  file for the changes made in the "ai" or "docs" submodules.

  Returns:
    Error code of `PermissionError`.
  """
  error_code = 0
  changelog = ai_logs.Changelog(_BASE_AI_REPO)
  try:
    changelog.write_changelog_md(
      'CHANGELOG.md', follow=(
        _BASE_AI_MODULE_PATH,
        _BASE_AI_DOCS_PATH,
      )
    )
  except PermissionError as exc:
    error_code = exc.errno
  return error_code


def Main(namespace: argparse.Namespace) -> int:
  """Takes actions according to the commands that are given over the
  command-line.

  Args:
    namespace: Stores the commands given over the command-line.
  """
  if namespace.generate_docs:
    return GenerateAIDocs()
  if namespace.generate_logs:
    return GenerateAILogs()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog=(
      'Post an issue at https://github.com/joshiayush/ai/issues '
      'for any modification in this program.'
    )
  )
  parser.add_argument(
    '--generate-docs',
    action='store_true',
    help='Triggers the action of generating documents from IPython Notebooks.'
  )
  parser.add_argument(
    '--generate-logs',
    action='store_true',
    help='Generates changelogs for the project.'
  )
  sys.exit(Main(parser.parse_args()))
