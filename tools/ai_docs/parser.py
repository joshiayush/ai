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

      Args:
        match: The matched math block, which is a re.Match object.

      Returns:
        The math block surrounded by ```math and ``` or the original
        match if it is not a multi-line block.
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
      # FIXME: Before joining the output keep in mind that not only 'text'
      # types are available in 'outputs' but also errors, and warnings may
      # be present in the 'outputs' causing 'KeyError' in case of the
      # absesnce of 'text' key.
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

  Splits the given markdown at each level one heading representing a topic and
  saves its content inside of a separate markdown file.

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

  if not os.path.exists(os.fspath(ipynb_docs_dir)):
    os.makedirs(ipynb_docs_dir)
  if not os.access(os.fspath(ipynb_docs_dir), os.W_OK):
    raise PermissionError(f'Cannot write directory {ipynb_docs_dir}')

  def _FindLevelOneHeadings(markdown: str) -> list[str]:
    pattern = r'(?<!`)(?<!#)# .*(?<!`)'
    code_block_pattern = r'```[\s\S]*?```'
    code_inline_pattern = r'`[^`\n]+?`'

    code_pattern = f'({code_block_pattern})|({code_inline_pattern})'
    markdown = re.sub(code_pattern, '', markdown)

    level_one_headings = []
    for match in re.finditer(pattern, markdown):
      level_one_headings.append(match.group(0))

    return level_one_headings

  # Find all level 1 headings that are not a part of code blocks.
  topics = _FindLevelOneHeadings(markdown)

  for i in range(len(topics)):
    current_topic_idx = i
    next_topic_idx = current_topic_idx + 1
    if next_topic_idx >= len(topics):
      topic_content = markdown[markdown.find(topics[current_topic_idx]) +
                               len(topics[current_topic_idx]):]
    else:
      topic_content = markdown[markdown.find(topics[current_topic_idx]) +
                               len(topics[current_topic_idx]):markdown.
                               find(topics[next_topic_idx])]
    markdown_ = f'{topics[i]}{topic_content}'

    ipynb_docs_file = '-'.join(topics[i].split(' ')[1:])
    ipynb_docs_file += '.md'
    with open(os.fspath(pathlib.Path(ipynb_docs_dir) / ipynb_docs_file),
              mode='w',
              encoding='utf-8') as docs:
      docs.write(markdown_)


# @TODO: "GenerateTableOfContents()" is deprecated, this function will be
# removed in the future commits. Refer to the "ai_html" module instead.
def GenerateTableOfContents(base_docs_dir: str | pathlib.Path) -> None:
  """Generates a table of contents for a base readme file based on the contents
  of the `docs` directory. The table of contents will include links to all
  markdown files in the `docs` directory, as well as any subdirectories, and
  will organize them by directory.

  Args:
    base_docs_dir (str | pathlib.Path): The base directory of the `docs`
                                        directory.

  Returns:
    str: A string representation of the table of contents.
  """
  toc = ''
  for dirpath, dirnames, filenames in os.walk(base_docs_dir):
    # Ignore hidden directories
    dirnames[:] = [d for d in dirnames if not d.startswith('.')]

    marked_ctype = False
    # Generate links for all files in the current directory
    for filename in filenames:
      if filename.endswith('.md'):
        filepath = os.path.join(dirpath, filename)
        link = os.path.relpath(filepath, base_docs_dir).replace('\\', '/')
        if marked_ctype is False:
          toc += f'- {link.split("/")[0][0].upper() + link.split("/")[0][1:]}\n'
          marked_ctype = True
        link = '/' + os.path.join('docs', link)
        filename = ' '.join(filename[:-3].split('-'))
        toc += f'  - [{filename}]({link})\n'

  return toc
