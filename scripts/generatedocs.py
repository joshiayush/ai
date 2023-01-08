#!/usr/bin/python3

"""Generate markdown files from IPython Notebooks."""

from __future__ import annotations

import os
import sys
import json
import codecs
import pathlib

_BASE_AI_DIR = 'ai'
_BASE_DOCS_DIR = 'docs'
_GLOB_EVERY_FILE_AND_DIR_REGEX = '**'

# There are a few latex icon that GitHub couldn't generate with the
# normal latex code therefore, we need to manually replace all of
# them with their corresponding icons.
_LATEX_CODE_TO_ICON = {'\\Rightarrow': '⇒'}

_LATEX_BLOCK = '$'
_LATEX_CBLOCK = _LATEX_BLOCK * 2

# The "Sundown" parser is unable to parse a few latex syntax like
# "$\begin{...}...\end{...}$", but if we replace the dollar "$" with
# the "math" block the "Sundown" parser may work for some syntax.
#
# So we must replace the above syntax with
# ```math\begin{cases}...\end{cases}```
_LATEX_LINEAR_EQUATION_BLOCKS = ['cases', 'pmatrix', 'bmatrix']


def GetFileByExtensionUnderGivenDirectory(
    file_ext: str,
    dir: str | pathlib.Path,  # pylint: disable=redefined-builtin
    *,
    recursive: bool = False  # pylint: disable=unused-argument
) -> list[pathlib.Path]:
  # @TODO: Implement a non-recursive routine.
  for file in pathlib.Path(dir).glob(  # pylint: disable=redefined-outer-name
      f'{_GLOB_EVERY_FILE_AND_DIR_REGEX}/*.{file_ext}'):
    yield file


def ReplaceLatexCodeToIcon(markdown: str) -> str:
  for latex_code, latex_icon in _LATEX_CODE_TO_ICON.items():
    markdown = markdown.replace(latex_code, latex_icon)
  return markdown


def ReplaceLatexBlockWithMath(markdown: str) -> str:

  def ReplaceBeginningBlock(latex_block: str, block_name: str,
                            markdown: str) -> str:
    old_block_syntax = f'{latex_block}\n\\begin{{{block_name}}}'
    new_block_syntax = f"""```math\n\\begin{{{block_name}}}"""
    return markdown.replace(old_block_syntax, new_block_syntax)

  def ReplaceEndBlock(latex_block: str, block_name: str, markdown: str) -> str:
    old_block_syntax = f'\\end{{{block_name}}}\n{latex_block}'
    new_block_syntax = f"""\\end{{{block_name}}}\n```"""
    return markdown.replace(old_block_syntax, new_block_syntax)

  # Replace the beginning and end blocks with the beginning and end
  # "math" blocks for both left-aligned and center-aligned latex blocks.
  for block_name in _LATEX_LINEAR_EQUATION_BLOCKS:
    for latex_block in [_LATEX_BLOCK, _LATEX_CBLOCK]:
      markdown = ReplaceBeginningBlock(latex_block, block_name, markdown)
      markdown = ReplaceEndBlock(latex_block, block_name, markdown)

  return markdown


def ReadIPythonNotebookToMarkdown(file_path: str | pathlib.Path) -> str:
  source = None
  with codecs.open(os.fspath(file_path), 'r') as file:  # pylint: disable=redefined-outer-name
    source = file.read()

  markdown = ''  # pylint: disable=redefined-outer-name
  source = json.loads(source)
  for cell in source['cells']:
    if cell['cell_type'] == 'markdown':
      markdown += ''.join(cell['source'])
    elif cell['cell_type'] == 'code':
      markdown += """```python"""
      markdown += '\n'
      markdown += ''.join(cell['source'])
      markdown += '\n'
      markdown += """```"""
      if cell['outputs']:
        markdown += '\n' * 2
        markdown += '### Output\n'
        for output in cell['outputs']:
          markdown += '\n'
          markdown += """```"""
          markdown += '\n'
          markdown += ''.join(output['text']).strip()
          markdown += '\n'
          markdown += """```"""
    markdown += '\n' * 2
  return ReplaceLatexBlockWithMath(ReplaceLatexCodeToIcon(markdown.strip()))


def GenerateDocs(base_docs_dir: str | pathlib.Path,
                 ipynb_file_path: str | pathlib.Path, markdown: str) -> None:
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


def Main() -> int:
  error = 0
  base_docs_dir = pathlib.Path(os.getcwd()) / _BASE_DOCS_DIR
  for file in GetFileByExtensionUnderGivenDirectory(
      'ipynb', os.fspath(pathlib.Path(os.getcwd()) / _BASE_AI_DIR)):
    try:
      GenerateDocs(base_docs_dir, file, ReadIPythonNotebookToMarkdown(file))
    except PermissionError:
      error += 1
  return error


if __name__ == '__main__':
  sys.exit(Main())
