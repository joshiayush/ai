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

"""Exporting public APIs to generate markdown documents from IPython
Notebooks.
"""

from .parser import (ReadIPythonNotebookToMarkdown, GenerateDocs,
                     GenerateTableOfContents)

__readme__ = """# Artificial Intelligence Guide

These are the source files for the guide on Artificial Intelligence.

To contribute to the Artificial Intelligence Guide, please read the
[style guide](https://www.tensorflow.org/community/contribute/docs_style).
"""
