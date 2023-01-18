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
"""Parses the IPython Notebooks and makes them compliant to GitHub's
markdown renderer.
"""

from __future__ import annotations

import re
import os
import json
import codecs
import pathlib

# There are a few latex icon that GitHub couldn't generate with the
# normal latex code therefore, we need to manually replace all of
# them with their corresponding icons.
_LATEX_CODE_TO_ICON = {'\\Rightarrow': '⇒'}

# Left-aligned Latex math block.
_LATEX_BLOCK = '$'
# Center-aligned Latex math block.
_LATEX_CBLOCK = _LATEX_BLOCK * 2


def _ReplaceLatexCodeToIcon(markdown: str) -> str:
  """Replaces Latex special code words directly with their corresponding
  icons.

  Since, GitHub uses "Sundown" which is not able to parse some of the Latex
  syntax since it's been made and the issues with Sundown parsing are still
  persistent. So until GitHub fixes that from their side we have to manually
  replace the special words with their corresponding icons.

  Args:
    markdown: The markdown generated from the IPython Notebooks.

  Returns:
    Markdown with replaced latex codes.
  """
  for latex_code, latex_icon in _LATEX_CODE_TO_ICON.items():
    markdown = markdown.replace(latex_code, latex_icon)
  return markdown


def _ReplaceLatexBlockWithMath(markdown: str) -> str:
  """Replaces Latex typical math block with the GitHub's "math" block.

  Since, GitHub use "Sundown" to render markdown files which is currently
  not able to render rich text for few Latex syntax. So, we replace the
  Latex syntax with their corresponding GitHub "math" blocks to have them
  rendered properly.

  Args:
      markdown (str): The markdown generated from the IPython Notebooks.

  Returns:
      str: Markdown with replaced Latex blocks.
    """

  # @TODO: Merge function AddMathCodeFormatting and _ReplaceLatexBlockWithMath
  def AddMathCodeFormatting(markdown):
    """Find all occurrences of the pattern $$...$$ or $...$ in the input string
    where "..." is one or more characters and replace them with the string
    "```math...```" if it contains a newline.

    Args:
        markdown (str): The markdown generated from the IPython Notebooks.

    Returns:
        str: Markdown with replaced blocks.
    """

    def repl(match):  # pylint: disable=invalid-name
      """Check whether the match contains a newline character or not,
      and if it does, it replaces it with the math code format, otherwise
      it returns the original match.
      """
      if '\n' in match.group(1):
        return '```math' + match.group(1) + '```'
      else:
        return match.group(0)

    markdown = re.sub(r'\$\$((?:(?!\$\$).)*)\$\$',
                      repl,
                      markdown,
                      flags=re.DOTALL)
    return re.sub(r'\$((?:(?!\$).)*)\$', repl, markdown, flags=re.DOTALL)

  return AddMathCodeFormatting(markdown)


def ReadIPythonNotebookToMarkdown(file_path: str | pathlib.Path) -> str:
  """Converts the IPython Notebooks into markdown files.

  Converts the IPython Notebooks into markdown files by joining each cell
  together based on the `cell_type` property.

  Args:
    file_path: IPython's Notebook path.

  Returns:
    Markdown generated from IPython Notebooks.
  """
  with codecs.open(os.fspath(file_path), 'r') as file:
    source = file.read()

  markdown = ''
  source = json.loads(source)
  for cell in source['cells']:
    if cell['cell_type'] == 'markdown':
      markdown += ''.join(cell['source'])
    elif cell['cell_type'] == 'code':
      # We assume the code cell to be a "Python" code cell.
      # @TODO: Identify the programming language and wrap the code around
      # its code block type.
      markdown += '\n'.join(
          ["""```python""", ''.join(cell['source']), """```"""])
      if cell['outputs']:
        markdown += '\n' * 2
        markdown += '### Output\n'
        for output in cell['outputs']:
          markdown += '\n'.join(
              ['\n', """```""", ''.join(output['text']).strip(), """```"""])
    markdown += '\n' * 2

  # Return the replaced version of old Latex style markdown with the GitHub
  # Sundown style markdown.
  return _ReplaceLatexBlockWithMath(_ReplaceLatexCodeToIcon(markdown.strip()))


def GenerateDocs(base_docs_dir: str | pathlib.Path,
                 ipynb_file_path: str | pathlib.Path, markdown: str) -> None:
  """Generates markdown documents compliant to GitHub's markdown renderer.

  Args:
    base_docs_dir:   Base `docs` directory to store the generated markdown
                     files in.
    ipynb_file_path: IPython Notebooks file path.
    markdown:        The markdown to write to the corresponding markdown
                     file of the IPython Notebook.
  """
  # Take out the "head" component, because that's where we would want
  # to store our generated document file.
  ipynb_docs_dir, ipynb_docs_file = os.path.split(ipynb_file_path)
  ipynb_docs_dir = pathlib.Path(base_docs_dir) / os.fspath(
      ipynb_docs_dir)[len(os.fspath(base_docs_dir)) - 1:]
  ipynb_docs_file, _ = os.path.splitext(ipynb_docs_file)
  ipynb_docs_file += '.md'

  if not os.path.exists(os.fspath(ipynb_docs_dir)):
    os.makedirs(ipynb_docs_dir)
  if not os.access(os.fspath(ipynb_docs_dir), os.W_OK):
    raise PermissionError(f'Cannot write directory {ipynb_docs_dir}')

  with open(os.fspath(pathlib.Path(ipynb_docs_dir) / ipynb_docs_file),
            mode='w',
            encoding='utf-8') as docs:
    docs.write(markdown)
