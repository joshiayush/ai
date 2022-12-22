# pylint: disable=missing-module-docstring

#!/usr/bin/python3

from __future__ import annotations

import os
import bs4
import sys
import json
import codecs
import pathlib
import difflib
import textblob
import markdown
import argparse

_GLOB_EVERY_FILE_AND_DIR_REGEX = '**'


def GetFileByExtensionUnderGivenDirectory(
    file_ext: str,
    dir: str,  # pylint: disable=redefined-builtin
    *,
    recursive: bool = False  # pylint: disable=unused-argument
) -> list[pathlib.Path]:
  # @TODO: Implement a non-recursive routine.
  for file in pathlib.Path(dir).glob(  # pylint: disable=redefined-outer-name
      f'{_GLOB_EVERY_FILE_AND_DIR_REGEX}/*.{file_ext}'):
    yield file


def ReadIPythonNotebook(file_path: pathlib.Path) -> str:
  source = None
  with codecs.open(str(file_path), 'r') as file:  # pylint: disable=redefined-outer-name
    source = file.read()

  markdown = ''  # pylint: disable=redefined-outer-name
  source = json.loads(source)
  for cell in source['cells']:
    for src in cell['source']:
      markdown += src

      # Append a newline character at the end of each individual line because
      # each individual markdown cell of the IPython Notebook does not
      # necessarily ends with a line-feed.
      if not src.endswith('\n'):
        markdown += '\n'
    markdown += '\n'
  return markdown


def ConvertMarkdownToText(html: str) -> str:
  return ''.join(
      bs4.BeautifulSoup(markdown.markdown(html),
                        'html.parser').find_all(text=True))


class SpellCheck:  # pylint: disable=missing-class-docstring

  def __init__(self, data: str) -> None:
    self.data = data

  def _list_words(self) -> list[str]:
    words = []
    lines = self.data.split('\n')
    for line in lines:
      words = [
          *words,
          *line.split(' '),
          '\n'  # We also include the line-feed as a word.
      ]
    return words

  def correct(self) -> str:
    corrected_words = []

    words = self._list_words()
    for word in words:
      corrected_words = [
          *corrected_words,
          # Only if the confidence byte is less than 0.5 we use
          # TextBlob.correct().
          str(textblob.TextBlob(word).correct())
          if textblob.Word(word).spellcheck()[0][1] < 0.5 else word
      ]
    return ' '.join(corrected_words)


def GenerateDiff(acutal: list[str], expected: list[str]) -> None:
  return difflib.unified_diff(acutal, expected)


def SpellChecker(namespace: argparse.Namespace) -> int:
  errors = 0
  directory = os.path.abspath(namespace.dir)
  for file in GetFileByExtensionUnderGivenDirectory(namespace.ext, directory):
    text = ConvertMarkdownToText(ReadIPythonNotebook(file))
    corrected_text = SpellCheck(text).correct()

    if text == corrected_text:
      continue
    errors += 1
    sys.stdout.writelines(GenerateDiff(text, corrected_text))
  return errors


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Checks the spelling mistakes of a file using NLTK.',
      epilog=('Post an issue at https://github.com/joshiayush/ai/issues '
              'for any modification in this program.'))
  parser.add_argument(
      '--dir',
      type=str,
      default='.',
      metavar='DIRECTORY',
      help='Directory to look for the file in (defaults to current directory).')
  parser.add_argument(
      '--ext',
      type=str,
      default='ipynb',
      metavar='EXTENSION',
      help='Limit files to the given extension (defaults to IPython Notebook).')
  sys.exit(SpellChecker(parser.parse_args()))
