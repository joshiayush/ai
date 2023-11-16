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
"""A changelog generator for `ai`."""

from __future__ import annotations

from typing import Union

import re
import os
import git
import copy
import pathlib

from datetime import date


class Changelog(git.Git):
  """Generates changelogs for `ai`."""
  git_log_fmt = '%s ([#%h](https://www.github.com/joshiayush/ai/commit/%h))'

  def __init__(self, working_dir: Union[str, pathlib.Path]):
    super().__init__(working_dir)
    self.changelog = dict()
    self.date = date.today()

  def _format_git_logs(self) -> str:
    """Formats git logs for writing into the changelog file."""
    changelog = '# Changelog'
    changelog += '\n' * 2
    for key in self.changelog:

      # Keys at the first level were the follow paths given during log
      # generation
      changelog += f'## {key} — {self.date}'
      changelog += '\n' * 2

      for git_log_t in self.changelog[key]:
        if len(self.changelog[key][git_log_t]) == 0:
          continue

        changelog += f'### {git_log_t.capitalize()}'
        changelog += '\n' * 2
        changelog += '\n'.join(
          map(lambda log: f'- {log}', self.changelog[key][git_log_t])
        )
        changelog += '\n' * 2
    return changelog

  def _get_git_logs(self, pretty: str,
                    follow: Union[str, pathlib.Path]) -> dict[str, set]:
    """Returns a dictionary containing the log records for every type of change.

    The log records are returned in a form of set to later subtract already
    published logs from it.

    Args:
      pretty: A prettified string format for the log statement.
      follow: A path for which the logs are to be generated. For example, if
        given `docs/`, it means follow the changes made to this directory only.

    Returns:
      A dictionary containing the log records for every type of change.
    """
    if follow:
      git_logs = self.log(f'--pretty={pretty}', '--follow', follow)
    else:
      git_logs = self.log(f'--pretty={pretty}')
    git_logs = git_logs.split('\n')

    git_logs_dict = dict()
    for git_log_t in (
      'added',
      'changed',
      'deprecated',
      'fixed',
      'removed',
      'security',
      'yanked',
    ):
      git_logs_dict.update(
        {
          f'{git_log_t}':
          set(
            filter(
              lambda changelog: changelog is not None,
              filter(
                lambda changelog: changelog.capitalize()
                if git_log_t == changelog.split()[0].lower() else None,
                git_logs
              )
            )
          )
        }
      )

    return git_logs_dict

  def read_changelog_md(self, fpath: Union[str, pathlib.Path]) -> str:
    """Returns the contents of the changelog file; omitting the first line.

    The first line is just `# CHANGELOG` which does not contribute to the actual
    values that needs to be parsed before updating the changelog file.

    Args:
      fpath: Path to the changelog file.

    Returns:
      The changelogs.
    """
    changelog = ''
    with open(fpath, mode='r', encoding='utf-8') as f:
      changelog = ''.join(f.readlines()[1:])
    return changelog

  def write_changelog_md(
    self, fpath: Union[str, pathlib.Path],
    follow: tuple[dict[str, tuple[Union[str, pathlib.Path]]]]
  ) -> None:
    """Writes the newly added logs to the changelog file.

    We parse the logs from the git logs and the changelog file into `set` to
    later apply a `difference` operation over these two sets obtaining the logs
    that are in the git logs but not in the changelog file, thus omitting the
    already added logs.

    Args:
      fpath: Writable changelog file path.
      follow: A path for which the logs are to be generated. For example, if
        given `docs/`, it means follow the changes made to this directory only.
    """
    for f in follow:
      for key, paths in f.items():
        for path in paths:
          git_logs = self._get_git_logs(self.git_log_fmt, follow=path)
          if key not in self.changelog:
            self.changelog[key] = git_logs
          else:
            for git_log_t in git_logs:
              self.changelog[key][git_log_t].update(git_logs[git_log_t])

    old_changelog = self.read_changelog_md(fpath)
    old_changelog = self.parse_changelog_md(old_changelog)
    tmp_changelog = dict()
    for k in old_changelog:
      tmp_changelog[k] = dict()
      for date in old_changelog[k]:
        for git_log_t in old_changelog[k][date]:
          if git_log_t not in tmp_changelog[k]:
            tmp_changelog[k][git_log_t] = old_changelog[k][date][git_log_t]
          else:
            tmp_changelog[k][git_log_t].update(
              old_changelog[k][date][git_log_t]
            )
    old_changelog = copy.deepcopy(tmp_changelog)
    del tmp_changelog

    if old_changelog:
      for k in self.changelog:
        for git_log_t in self.changelog[k]:
          if git_log_t in self.changelog[k] and git_log_t in old_changelog[k]:
            self.changelog[k][git_log_t].difference_update(
              old_changelog[k][git_log_t]
            )

    with open(fpath, mode='r', encoding='utf-8') as f:
      old_changelog = ''.join(f.readlines()[1:])

    with open(fpath, mode='w', encoding='utf-8') as f:
      f.write(self._format_git_logs())

    with open(fpath, mode='a', encoding='utf-8') as f:
      f.write(old_changelog)

  def parse_changelog_md(self,
                         changelog: str) -> dict[str, dict[str, set[str]]]:
    """Parse the changelog file to keep a list of all the logs recorded.

    The logs are parsed in the form of a dictionary containing the follow path
    mapped to each version of their changelogs.

    Args:
      changelog: A list containing individual lines of the changelog file.

    Returns:
      The changelog dictionary containing the follow path mapped to each version
      of their changelogs.
    """
    changelog_dict = dict()
    cur_follow = None
    cur_date = None
    cur_section = None

    date_pattern = re.compile(r'##\s(\w+)\s—\s(\d{4}-\d{2}-\d{2})')
    section_pattern = re.compile(r'###\s(\w+)')
    entry_pattern = re.compile(r'-\s(.*)')

    for line in changelog.split('\n'):
      date_match = date_pattern.match(line)
      section_match = section_pattern.match(line)
      entry_match = entry_pattern.match(line)

      if date_match:
        cur_date = date_match.group(2)
        cur_follow = date_match.group(1)
        if cur_follow not in changelog_dict:
          changelog_dict[cur_follow] = dict()
        changelog_dict[cur_follow][cur_date] = dict()
        cur_section = None
      elif section_match:
        cur_section = section_match.group(1)
        cur_section = cur_section.lower()
        changelog_dict[cur_follow][cur_date][cur_section] = set()
      elif entry_match and cur_section:
        changelog_dict[cur_follow][cur_date][cur_section].update(
          [entry_match.group(1)]
        )

    return changelog_dict
