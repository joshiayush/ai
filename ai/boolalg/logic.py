# Copyright 2023 The AI Authors. All Rights Reserved.
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
# pylint: disable=too-many-function-args, invalid-name, missing-module-docstring
# pylint: disable=missing-class-docstring

import itertools


class _Sentence():
  """An abstract model used for implementing symbols and logical connectives
  used in Propositional Logic.

  .. note::

    See `Symbol` for more details.
  """
  def evaluate(self, model: dict[str, bool]) -> bool:
    """Evaluates the value of a symbol in a propositional logic (PL)."""
    raise NotImplementedError("nothing to evaluate")

  def formula(self) -> str:
    """Returns string formula representing logical sentence."""
    return ""

  def symbols(self) -> str:
    """Returns a set of all symbols in the logical sentence."""
    return set()

  @classmethod
  def validate(cls, sentence: 'Sentence'):
    """Validates if the given sentence is actually a `Sentence` instance.

    Raises:
      TypeError - In case of a non-logical sentence.
    """
    if not isinstance(sentence, _Sentence):
      raise TypeError("must be a logical sentence")

  @classmethod
  def parenthesize(cls, s: str) -> str:
    """Parenthesizes an expression if not already parenthesized.

    Args:
      s: Expression to parenthesize.

    Returns:
      Parenthesized expression.
    """
    def balanced(s: str) -> bool:
      """Checks if a string has balanced parentheses.

      Args:
        s: Expression to check for balanced parentheses.

      Returns:
        `True` if expression contains balanced parentheses, `False` otherwise.
      """
      count = 0
      for c in s:
        if c == "(":
          count += 1
        elif c == ")":
          if count <= 0:
            return False
          count -= 1
      return count == 0

    # Empty or alpha strings are always considered to be balanced.
    if not len(s) or s.isalpha(
    ) or (s[0] == "(" and s[-1] == ")" and balanced(s[1:-1])):
      return s
    else:
      return f"({s})"


class Symbol(_Sentence):
  """`Symbol` class creates a symbol for a propositional logic (PL).

  A proposition is a declarative statement which is either true or false. The
  symbols used in propositional logic determines whether the logic holds `true`
  or turns out to be `false`.

  #### Examples

  >>> from logic import (Symbol, And)
  >>> P = Symbol('P')
  >>> Q = Symbol('Q')
  >>> knowledge = And(P, Q)
  >>> knowledge.formula()
  'P ∧ Q'
  """
  def __init__(self, name: str):
    """Constructs a `Symbol` instance.

    A symbol in propositional logic (PL) either holds a truth value or a false
    value, based on the combination of logical operations these values are then
    transformed to a result.

    .. note::

      You can use letters, words, digits, or even special characters to contruct
      symbols as long as they hold some meaning for the viewer.

    Args:
      name: `Symbol` name; can be anything including digits, words, letters, or
        special characters.

    #### Examples

    >>> P = Symbol('P')
    >>> Q = Symbol('Q')
    >>> R = Symbol('R')
    """
    self._name = name

  def __eq__(self, other: 'Symbol') -> bool:
    """Compares `self` with the other."""
    return isinstance(other, Symbol) and self._name == other._name

  def __hash__(self) -> int:
    """Returns the hash of the current `Symbol`."""
    return hash(("symbol", self._name))

  def __repr__(self) -> str:
    """Returns the name of the symbol."""
    return self._name

  def evaluate(self, model: dict[str, bool]) -> bool:
    """Evaluates the value of a symbol in a propositional logic (PL).

    Evaluating a model means evaluating the value of each symbol in the
    propositional logic (PL) which either holds `True` or `False`.

    For example, for a symbol :math:`P \\implies \\mathrm{True}` in a
    propositional logic (PL) —

    .. math::

      ¬P \\implies \\mathrm{False}

    Args:
      model: A propositional logic model mapping from the symbol name to its
        truth or false value.

    Returns:
      The evaluated model of the symbol.

    #### Examples

    >>> from ai import Symbol, Not
    >>> P = Symbol('P')
    >>> knowledge = Not(P)
    >>> model_true[str(P)] = True
    >>> knowledge.evaluate(model_true)
    False
    """
    try:
      return bool(model[self._name])
    except KeyError:
      raise EvaluationException(f"variable {self._name} not in model")

  def formula(self) -> str:
    """Returns the name of the symbol to be used in the formula.

    This function is later used to combine the operators with the operands i.e.,
    the symbols which are a `Symbol` instance. This function returns the string
    representation of the symbols which is then later joined with the operators
    to complete the propositional logic (PL).

    Returns:
      String representation of the current symbol.

    #### Examples

    >>> P = Symbol('P')
    >>> P.formula()
    'P'
    """
    return self._name

  def symbols(self) -> str:
    """Returns a set containing the name of the symbol."""
    return {self._name}


class Not(_Sentence):
  def __init__(self, operand):
    _Sentence.validate(operand)
    self.operand = operand

  def __eq__(self, other):
    return isinstance(other, Not) and self.operand == other.operand

  def __hash__(self):
    return hash(("not", hash(self.operand)))

  def __repr__(self):
    return f"Not({self.operand})"

  def evaluate(self, model):
    return not self.operand.evaluate(model)

  def formula(self):
    return "¬" + _Sentence.parenthesize(self.operand.formula())

  def symbols(self):
    return self.operand.symbols()


class And(_Sentence):
  def __init__(self, *conjuncts):
    for conjunct in conjuncts:
      _Sentence.validate(conjunct)
    self.conjuncts = list(conjuncts)

  def __eq__(self, other):
    return isinstance(other, And) and self.conjuncts == other.conjuncts

  def __hash__(self):
    return hash(("and", tuple(hash(conjunct) for conjunct in self.conjuncts)))

  def __repr__(self):
    conjunctions = ", ".join([str(conjunct) for conjunct in self.conjuncts])
    return f"And({conjunctions})"

  def add(self, conjunct):
    _Sentence.validate(conjunct)
    self.conjuncts.append(conjunct)

  def evaluate(self, model):
    return all(conjunct.evaluate(model) for conjunct in self.conjuncts)

  def formula(self):
    if len(self.conjuncts) == 1:
      return self.conjuncts[0].formula()
    return " ∧ ".join(
      [
        _Sentence.parenthesize(conjunct.formula())
        for conjunct in self.conjuncts
      ]
    )

  def symbols(self):
    return set.union(*[conjunct.symbols() for conjunct in self.conjuncts])


class Or(_Sentence):
  def __init__(self, *disjuncts):
    for disjunct in disjuncts:
      _Sentence.validate(disjunct)
    self.disjuncts = list(disjuncts)

  def __eq__(self, other):
    return isinstance(other, Or) and self.disjuncts == other.disjuncts

  def __hash__(self):
    return hash(("or", tuple(hash(disjunct) for disjunct in self.disjuncts)))

  def __repr__(self):
    disjuncts = ", ".join([str(disjunct) for disjunct in self.disjuncts])
    return f"Or({disjuncts})"

  def evaluate(self, model):
    return any(disjunct.evaluate(model) for disjunct in self.disjuncts)

  def formula(self):
    if len(self.disjuncts) == 1:
      return self.disjuncts[0].formula()
    return " ∨ ".join(
      [
        _Sentence.parenthesize(disjunct.formula())
        for disjunct in self.disjuncts
      ]
    )

  def symbols(self):
    return set.union(*[disjunct.symbols() for disjunct in self.disjuncts])


class Implication(_Sentence):
  def __init__(self, antecedent, consequent):
    _Sentence.validate(antecedent)
    _Sentence.validate(consequent)
    self.antecedent = antecedent
    self.consequent = consequent

  def __eq__(self, other):
    return (
      isinstance(other, Implication) and
      self.antecedent == other.antecedent and
      self.consequent == other.consequent
    )

  def __hash__(self):
    return hash(("implies", hash(self.antecedent), hash(self.consequent)))

  def __repr__(self):
    return f"Implication({self.antecedent}, {self.consequent})"

  def evaluate(self, model):
    return (
      (not self.antecedent.evaluate(model)) or self.consequent.evaluate(model)
    )

  def formula(self):
    antecedent = _Sentence.parenthesize(self.antecedent.formula())
    consequent = _Sentence.parenthesize(self.consequent.formula())
    return f"{antecedent} => {consequent}"

  def symbols(self):
    return set.union(self.antecedent.symbols(), self.consequent.symbols())


class Biconditional(_Sentence):
  def __init__(self, left, right):
    _Sentence.validate(left)
    _Sentence.validate(right)
    self.left = left
    self.right = right

  def __eq__(self, other):
    return (
      isinstance(other, Biconditional) and self.left == other.left and
      self.right == other.right
    )

  def __hash__(self):
    return hash(("biconditional", hash(self.left), hash(self.right)))

  def __repr__(self):
    return f"Biconditional({self.left}, {self.right})"

  def evaluate(self, model):
    return (
      (self.left.evaluate(model) and self.right.evaluate(model)) or
      (not self.left.evaluate(model) and not self.right.evaluate(model))
    )

  def formula(self):
    left = _Sentence.parenthesize(str(self.left))
    right = _Sentence.parenthesize(str(self.right))
    return f"{left} <=> {right}"

  def symbols(self):
    return set.union(self.left.symbols(), self.right.symbols())


def model_check(knowledge, query):
  """Checks if knowledge base entails query."""
  def check_all(knowledge, query, symbols, model):
    """Checks if knowledge base entails query, given a particular model."""

    # If model has an assignment for each symbol
    if not symbols:

      # If knowledge base is true in model, then query must also be true
      if knowledge.evaluate(model):
        return query.evaluate(model)
      return True
    else:

      # Choose one of the remaining unused symbols
      remaining = symbols.copy()
      p = remaining.pop()

      # Create a model where the symbol is true
      model_true = model.copy()
      model_true[p] = True

      # Create a model where the symbol is false
      model_false = model.copy()
      model_false[p] = False

      # Ensure entailment holds in both models
      return (
        check_all(knowledge, query, remaining, model_true) and
        check_all(knowledge, query, remaining, model_false)
      )

  # Get all symbols in both knowledge and query
  symbols = set.union(knowledge.symbols(), query.symbols())

  # Check that knowledge entails query
  return check_all(knowledge, query, symbols, dict())
