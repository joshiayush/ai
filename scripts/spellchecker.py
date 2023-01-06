# pylint: disable=missing-module-docstring

#!/usr/bin/python3

from __future__ import annotations

import os
import bs4
import sys
import json
import string
import codecs
import pathlib
import difflib
import textblob
import markdown
import argparse
import langdetect

from nltk import tokenize
from nltk.corpus import stopwords

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
      # Append a newline character at the end of each individual line
      # because each individual markdown cell of the IPython Notebook
      # does not necessarily ends with a line-feed.
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
      try:
        words = [*words, *tokenize.word_tokenize(line, 'english'), '\n']
      except LookupError:
        import nltk  # pylint: disable=import-outside-toplevel
        nltk.download('punkt')
        words = [*words, *tokenize.word_tokenize(line, 'english'), '\n']
    return words

  def _group_words_by_sentence(
      self, word_list: list[str]) -> list[tuple[str, list[str]]]:
    sentence = []
    sentence_list = []
    for word in word_list:
      sentence = [*sentence, word]
      if word == '\n' and len(sentence) > 1:
        sentence_list = [
            *sentence_list, (langdetect.detect(' '.join(sentence)), sentence)
        ]
        sentence = []
    return sentence_list

  def _check_if_stop_word(self, word: str) -> bool:
    try:
      if word in stopwords.words('english'):
        return True
    except LookupError:
      import nltk  # pylint: disable=import-outside-toplevel
      nltk.download('stopwords')

    if word in stopwords.words('english'):
      return True
    return False

  def correct(self) -> str:
    sentence_list = self._group_words_by_sentence(self._list_words())
    for sentence in sentence_list:
      if sentence[0] != 'en':
        continue
      for i in range(len(sentence[1])):
        if self._check_if_stop_word(sentence[1][i]) is True or sentence[1][
            i] in string.punctuation or sentence[1][
                i] in string.digits or sentence[1][i] in string.whitespace:
          continue
        # Only if the confidence byte is less than 0.5 we use
        # TextBlob.correct().
        if textblob.Word(sentence[1][i]).spellcheck()[0][1] < 0.5:
          sentence[1][i] = str(textblob.TextBlob(sentence[1][i]).correct())

    sentence_list_without_lang = []
    for sentence in sentence_list:
      sentence_list_without_lang = [*sentence_list_without_lang, *sentence[1]]
    return ' '.join(sentence_list_without_lang)


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
