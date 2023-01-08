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

"""A dedicated implementation of re-useable filesystem functions."""

from __future__ import annotations

import pathlib

# Glob search pattern to search for every file and directory.
_GLOB_EVERY_FILE_AND_DIR_REGEX = '**'


def GetFileByExtensionUnderGivenDirectory(
    file_ext: str,
    dir: str | pathlib.Path,  # pylint: disable=redefined-builtin
    *,
    recursive: bool = False  # pylint: disable=unused-argument
) -> pathlib.Path:
  """Returns a `Path` object representing a filesystem path.

  Only returns a list of `Path` objects for filesystem path representing
  files under the given `dir`.

  Args:
    file_ext:  The extension the filesystem path should ends with.
    dir:       The directory under which the files with the given extension
               should be searched for.
    recursive: When `True`, searches recursively under the given directory
               for files ending with the given extension.

  Yields:
    A `Path` instance representing a filesystem path.
  """

  # @TODO: Implement a non-recursive routine.
  for file in pathlib.Path(dir).glob(
      f'{_GLOB_EVERY_FILE_AND_DIR_REGEX}/*.{file_ext}'):
    yield file
