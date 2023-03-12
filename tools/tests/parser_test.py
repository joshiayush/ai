# pylint: disable=missing-module-docstring

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


def test_replace_latex_block_with_math_empty_string():
  markdown = ''
  expected_output = ''
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output  # pylint: disable=protected-access


def test_replace_latex_block_with_math_no_latex_block():
  markdown = 'This is a plain text with no latex block.'
  expected_output = markdown
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output  # pylint: disable=protected-access


def test_replace_latex_block_with_math_inline_latex_block():
  markdown = 'This is a text with an inline latex block $x+y=z$.'
  expected_output = markdown
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output  # pylint: disable=protected-access


def test_replace_latex_block_with_math_multiline_latex_block():
  markdown = """This is a text with a multiline latex block:
  $$
  x+y=z
  $$"""
  expected_output = """This is a text with a multiline latex block:
  ```math
  x+y=z
  ```"""
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output  # pylint: disable=protected-access


def test_replace_latex_block_with_math_multiple_latex_block():
  markdown = """This is a text with multiple latex block:
  $$
  x+y=z
  $$
  And inline block $x-y=z$."""
  expected_output = """This is a text with multiple latex block:
  ```math
  x+y=z
  ```
  And inline block $x-y=z$."""
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output  # pylint: disable=protected-access


def test_replace_latex_block_with_math_multiple_syntax():
  markdown = 'This is a text with multiple syntax: $$x+y=z$$ and $x-y=z$'
  expected_output = markdown
  assert parser._ReplaceLatexBlockWithMath(markdown) == expected_output  # pylint: disable=protected-access


@pytest.fixture
def ipynb_file_path():
  with tempfile.NamedTemporaryFile(suffix='.ipynb') as f:
    f.write(b'{"cells": []}')
    f.seek(0)
    yield pathlib.Path(f.name)


def test_generate_docs(ipynb_file_path):  # pylint: disable=redefined-outer-name
  base_docs_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
  markdown = '# Topic 1\nContent 1\n# Topic 2\nContent 2\n'
  parser.GenerateDocs(base_docs_dir.name, ipynb_file_path, markdown)

  docs_dir = pathlib.Path(base_docs_dir.name)
  assert docs_dir.exists()

  assert (docs_dir / 'Topic-1.md').exists()
  with open(docs_dir / 'Topic-1.md', encoding='utf-8') as f:
    assert f.read() == '# Topic 1\nContent 1\n'

  assert (docs_dir / 'Topic-2.md').exists()
  with open(docs_dir / 'Topic-2.md', encoding='utf-8') as f:
    assert f.read() == '# Topic 2\nContent 2\n'

  base_docs_dir.cleanup()


def test_generate_docs_with_code_blocks(ipynb_file_path):  # pylint: disable=redefined-outer-name
  base_docs_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
  markdown = """# Topic 1

Some content here.

```
# This should not be considered a heading
```

# Topic 2

More content here.

```python
import re
# Imports re module
```
"""

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
