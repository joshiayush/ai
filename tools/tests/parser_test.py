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
