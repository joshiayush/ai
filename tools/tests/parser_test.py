# pylint: disable=missing-module-docstring, protected-access
# pylint: disable=redefined-outer-name

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

import pytest
import pathlib
import tempfile

from ai_docs import parser


def test_replace_latex_code_with_icon():
  markdown = ''.join(['\\Rightarrow'])
  expected_output = ''.join(['⇒'])
  assert parser._ReplaceLatexCodeToIcon(markdown) == expected_output


def test_replace_latex_block_with_math_when_no_latex_is_present():
  markdown = ''
  expected_output = ''
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output

  markdown = 'No latex block'
  expected_output = markdown
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output


def test_replace_latex_block_with_math_when_latex_is_present():
  markdown = 'Inline latex block here $x+y=z$'
  expected_output = markdown
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output

  markdown = '\n'.join(['Multi-line latex block here', '$$', 'x+y=z', '$$'])
  expected_output = '\n'.join(
      ['Multi-line latex block here', '```math', 'x+y=z', '```'])
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output


def test_replace_latex_block_with_math_when_multiple_latex_is_present():
  markdown = '\n'.join([
      'Inline latex block here $x-y=z$.', 'Multi-line latex block here', '$$',
      'x+y=z', '$$'
  ])
  expected_output = '\n'.join([
      'Inline latex block here $x-y=z$.', 'Multi-line latex block here',
      '```math', 'x+y=z', '```'
  ])
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output

  markdown = '\n'.join([
      'Inline latex block here $x-y=z$', 'Multi-line latex block here',
      '$$x+y=z$$'
  ])
  expected_output = '\n'.join([
      'Inline latex block here $x-y=z$', 'Multi-line latex block here',
      '$$x+y=z$$'
  ])
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output


@pytest.fixture
def ipynb_file_path():
  with tempfile.NamedTemporaryFile(suffix='.ipynb') as f:
    f.write(b'{"cells": []}')
    f.seek(0)
    yield pathlib.Path(f.name)


def test_generate_docs_with_code_blocks(ipynb_file_path):
  base_docs_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
  markdown = '\n'.join([
      '# Topic 1', '', 'Some content here.', '', '```',
      '# This should not be considered a heading', '```', '', '# Topic 2', '',
      'More content here.', '', '```python', 'import re', '# Imports re module',
      '```'
  ])

  parser.GenerateDocs(base_docs_dir.name, ipynb_file_path, markdown)

  docs_dir = pathlib.Path(base_docs_dir.name)
  assert docs_dir.exists()

  assert (pathlib.Path(docs_dir) / 'Topic-1.md').exists()
  assert (pathlib.Path(docs_dir) / 'Topic-2.md').exists()

  assert not (pathlib.Path(docs_dir) /
              'This-should-not-be-considered-a-heading.md').exists()

  with open(docs_dir / 'Topic-1.md', 'r', encoding='utf-8') as f:
    topic1_content = f.read()
    assert topic1_content.startswith('# Topic 1\n\nSome content here.\n\n')

  with open(docs_dir / 'Topic-2.md', 'r', encoding='utf-8') as f:
    topic2_content = f.read()
    assert topic2_content.startswith('# Topic 2\n\nMore content here.\n')

  base_docs_dir.cleanup()
