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

from __future__ import annotations

from typing import Union

import os
import git
import pathlib


class AIGitLog(git.Git):
  class _AIChangeLogParser:
    def __init__(self, changelogs: list[str]):
      self.changelogs = changelogs

    def _filter_out_logs(self,
                         log_type: str,
                         *,
                         capitalize: bool = False) -> list[str]:
      filtered_logs = tuple()
      for changelog in self.changelogs:
        if log_type.lower() == changelog.split()[0].lower():
          filtered_logs = (
            *filtered_logs,
            changelog if capitalize is False else changelog[0].upper() +
            changelog[1:]
          )
      return filtered_logs

    def parse(self) -> dict[str, list]:
      self.added = self._filter_out_logs('added', capitalize=True)
      self.changed = self._filter_out_logs('changed', capitalize=True)
      self.deprecated = self._filter_out_logs('deprecated', capitalize=True)
      self.removed = self._filter_out_logs('removed', capitalize=True)
      self.fixed = self._filter_out_logs('fixed', capitalize=True)
      self.security = self._filter_out_logs('security', capitalize=True)
      self.yanked = self._filter_out_logs('yanked', capitalize=True)

      return {
        'added': self.added,
        'changed': self.changed,
        'deprecated': self.deprecated,
        'removed': self.removed,
        'fixed': self.fixed,
        'security': self.security,
        'yanked': self.yanked
      }

  def __init__(self, working_dir: Union[str, pathlib.Path]):
    super().__init__(working_dir)

  def _get_changelog_splitted(self,
                              pretty: str,
                              follow: str = None) -> list[str]:
    if follow:
      return self.log(f'--pretty={pretty}', '--follow', follow).split('\n')
    return self.log(f'--pretty={pretty}').split('\n')

  def get_changelog(self, pretty: str, follow: str = None) -> dict[str, tuple]:
    parser = AIGitLog._AIChangeLogParser(
      self._get_changelog_splitted(pretty, follow)
    )
    return parser.parse()

  def _prettify(self, changelog: list[str]) -> str:
    def write_for(heading: str) -> str:
      nonlocal changelog
      prettified = ''
      if changelog[heading]:
        added = []
        for log in changelog[heading]:
          added = [*added, f'- {log}']
        prettified += f'### {heading.capitalize()}'
        prettified += '\n\n'
        prettified += '\n'.join(added)
      return prettified

    return '\n'.join(
      [
        write_for('added'),
        write_for('changed'),
        write_for('deprecated'),
        write_for('fixed'),
        write_for('removed'),
        write_for('security'),
        write_for('yanked')
      ]
    )

  def parse_changelogs_for_all(self,
                               follow_path: tuple[str]) -> dict[str, tuple]:
    merged_changelog_dict = {
      'added': tuple(),
      'changed': tuple(),
      'deprecated': tuple(),
      'fixed': tuple(),
      'removed': tuple(),
      'security': tuple(),
      'yanked': tuple(),
    }
    for follow in follow_path:
      changelogs = self.get_changelog(
        '%s ([#%h](https://www.github.com/joshiayush/ai/commit/%h))',
        follow=follow
      )
      for key in merged_changelog_dict:
        merged_changelog_dict[key] = (
          *merged_changelog_dict[key], *changelogs[key]
        )

    for key in merged_changelog_dict:
      merged_changelog_dict[key] = tuple(set(merged_changelog_dict[key]))
    return self._prettify(merged_changelog_dict)

  def write_changelog(
    self, fpath: Union[str, pathlib.Path], changelog: str
  ) -> None:
    old_changelogs = None
    if os.access(os.fspath(fpath), os.F_OK):
      with open(fpath, mode='r', encoding='utf-8') as f:
        old_changelogs = ''.join(f.readlines()[1:])

    changelog = f'# CHANGELOG\n\n{changelog}'
    with open(fpath, mode='w', encoding='utf-8') as f:
      f.write(changelog)

    if old_changelogs:
      with open(fpath, mode='a', encoding='utf-8') as f:
        f.writelines(['\n', '\n', old_changelogs])
