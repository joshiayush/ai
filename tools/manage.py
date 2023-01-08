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

import os
import sys
import pathlib
import argparse

import ai_docs
import ai_utils

_BASE_AI_DIR = 'ai'
_BASE_DOCS_DIR = 'docs'


def GenerateAIDocs() -> int:
  """Completes the action of command "--generate-docs" by generating markdown
  files from the IPython Notebooks.

  Returns:
    Error code of `PermissionError`.
  """
  error_code = 0
  base_docs_dir = pathlib.Path(os.getcwd()) / _BASE_DOCS_DIR
  for file in ai_utils.GetFileByExtensionUnderGivenDirectory(
      'ipynb', os.fspath(pathlib.Path(os.getcwd()) / _BASE_AI_DIR)):
    try:
      ai_docs.GenerateDocs(base_docs_dir, file,
                           ai_docs.ReadIPythonNotebookToMarkdown(file))
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=__doc__,
      epilog=('Post an issue at https://github.com/joshiayush/ai/issues '
              'for any modification in this program.'))
  parser.add_argument(
      '--generate-docs',
      action='store_true',
      help='Triggers the action of generating documents from IPython Notebooks.'
  )
  sys.exit(Main(parser.parse_args()))
