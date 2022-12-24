#!/usr/bin/python3
"""A command-line utility program to manage the word database.

We group the word database by the initial letters in separate files. We may
think of using a database in future but for now we use multiple files to
store words.

We also sort the words lexicographically to maintain the order.
"""

from __future__ import annotations

import io
import os
import sys
import string
import pathlib
import argparse

_WORD_PUNCTUATION = "!@#$%^&*'-"

WordException = type('WordException', (Exception,), {})


def SortWordsLexicographically(file: str) -> None:
  if file is None:
    raise ValueError('Expected str or Path like object, received None')

  def WriteWordsToFile(file: io.TextIOWrapper, words: list[str]) -> None:
    file.seek(0)
    file.truncate()
    file.writelines(words)

  is_dir = os.path.isdir(file)
  if is_dir is False:
    with open(file, mode='a+', encoding='utf-8') as f:
      # Note the file is always opened using 'a+' mode opening it agian means
      # the file pointer is at the end-of-file so we must reset it.
      f.seek(0)
      words = f.readlines()
      words.sort()
      WriteWordsToFile(file, words)
    return

  for file_ in pathlib.Path(file).glob('*'):
    with open(file_, mode='a+', encoding='utf-8') as f:
      # Note the file is always opened using 'a+' mode opening it agian means
      # the file pointer is at the end-of-file so we must reset it.
      f.seek(0)
      words = f.readlines()
      words.sort()
      WriteWordsToFile(file, words)


def GroupWordsByLetters(file: str, dest: str) -> list[pathlib.Path]:
  if file is None or dest is None:
    raise ValueError('Expected str or Path like object, received None')

  file = os.path.abspath(file)
  if not os.path.exists(file) and os.path.isfile(file) and os.access(
      file, os.R_OK):
    raise WordException(
        'Check if the file is present or if it has read permissions')

  dest = os.path.abspath(dest)

  word_group = {}
  for letter in string.ascii_lowercase:
    word_group[letter] = []
  for character in _WORD_PUNCTUATION:
    word_group[character] = []
  for digit in string.digits:
    word_group[digit] = []

  lines = []
  with open(file=file, mode='r', encoding='utf-8') as f:
    lines = [*f.readlines()]

  for word in lines:
    word_group[word[0].lower()] = [*word_group[word[0].lower()], word]

  if not os.path.exists(dest):
    os.makedirs(dest)

  for key, value in word_group.items():
    with open(f'{dest}/{key}', mode='w', encoding='utf-8') as f:
      for word in value:
        f.write(word)


def Main(namespace: argparse.Namespace) -> int:
  errors = 0
  if namespace.group_words is True:
    try:
      GroupWordsByLetters(namespace.file, namespace.dest)
    except Exception as exc:  # pylint: disable=broad-except
      errors += 1
      sys.stderr.write(str(exc))
  elif namespace.sort_words is True:
    try:
      SortWordsLexicographically(namespace.file)
    except Exception as exc:  # pylint: disable=broad-except
      errors += 1
      sys.stderr.write(str(exc))
  return errors


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='A command-line utility to manage the word database.',
      epilog=('Post an issue at https://github.com/joshiayush/ai/issues '
              'for any modification in this program.'))
  parser.add_argument(
      '--group-words',
      action='store_true',
      help=('Executes the action of grouping the words in separate files by '
            'their starting letter.'))
  parser.add_argument('--sort-words',
                      action='store_true',
                      help='Sorts the words in a file lexicographically.')
  parser.add_argument('--file',
                      type=str,
                      default=None,
                      metavar='FILE',
                      help='File to perform the operation on.')
  parser.add_argument('--dest',
                      type=str,
                      default='.',
                      metavar='DIRECTORY',
                      help='Destination to generate resulting files in.')
  sys.exit(Main(parser.parse_args()))
