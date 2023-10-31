"""
Bash-style brace expansion
Copied from: https://github.com/trendels/braceexpand/blob/main/src/braceexpand/__init__.py
License: MIT
"""

import re
import string
from itertools import chain, product
from typing import Iterable, Iterator, Optional

__all__ = ["braceexpand", "alphabet", "UnbalancedBracesError"]


class UnbalancedBracesError(ValueError):
    pass


alphabet = string.ascii_uppercase + string.ascii_lowercase

int_range_re = re.compile(r"^(-?\d+)\.\.(-?\d+)(?:\.\.-?(\d+))?$")
char_range_re = re.compile(r"^([A-Za-z])\.\.([A-Za-z])(?:\.\.-?(\d+))?$")
escape_re = re.compile(r"\\(.)")


def braceexpand(pattern: str, escape: bool = True) -> Iterator[str]:
    """braceexpand(pattern) -> iterator over generated strings

    Returns an iterator over the strings resulting from brace expansion
    of pattern. This function implements Brace Expansion as described in
    bash(1), with the following limitations:

    * A pattern containing unbalanced braces will raise an
      UnbalancedBracesError exception. In bash, unbalanced braces will either
      be partly expanded or ignored.

    * A mixed-case character range like '{Z..a}' or '{a..Z}' will not
      include the characters '[]^_`' between 'Z' and 'a'.

    When escape is True (the default), characters in pattern can be
    prefixed with a backslash to cause them not to be interpreted as
    special characters for brace expansion (such as '{', '}', ',').
    To pass through a a literal backslash, double it ('\\\\').

    When escape is False, backslashes in pattern have no special
    meaning and will be preserved in the output.

    Examples:

    >>> from braceexpand import braceexpand

    # Integer range
    >>> list(braceexpand('item{1..3}'))
    ['item1', 'item2', 'item3']

    # Character range
    >>> list(braceexpand('{a..c}'))
    ['a', 'b', 'c']

    # Sequence
    >>> list(braceexpand('index.html{,.backup}'))
    ['index.html', 'index.html.backup']

    # Nested patterns
    >>> list(braceexpand('python{2.{5..7},3.{2,3}}'))
    ['python2.5', 'python2.6', 'python2.7', 'python3.2', 'python3.3']

    # Prefixing an integer with zero causes all numbers to be padded to
    # the same width.
    >>> list(braceexpand('{07..10}'))
    ['07', '08', '09', '10']

    # An optional increment can be specified for ranges.
    >>> list(braceexpand('{a..g..2}'))
    ['a', 'c', 'e', 'g']

    # Ranges can go in both directions.
    >>> list(braceexpand('{4..1}'))
    ['4', '3', '2', '1']

    # Numbers can be negative
    >>> list(braceexpand('{2..-1}'))
    ['2', '1', '0', '-1']

    # Unbalanced braces raise an exception.
    >>> list(braceexpand('{1{2,3}'))
    Traceback (most recent call last):
        ...
    UnbalancedBracesError: Unbalanced braces: '{1{2,3}'

    # By default, the backslash is the escape character.
    >>> list(braceexpand(r'{1\\{2,3}'))
    ['1{2', '3']

    # Setting 'escape' to False disables backslash escaping.
    >>> list(braceexpand(r'\\{1,2}', escape=False))
    ['\\\\1', '\\\\2']

    """
    return (
        escape_re.sub(r"\1", s) if escape else s for s in parse_pattern(pattern, escape)
    )


def parse_pattern(pattern: str, escape: bool) -> Iterator[str]:
    start = 0
    pos = 0
    bracketdepth = 0
    items: list[Iterable[str]] = []

    # print 'pattern:', pattern
    while pos < len(pattern):
        if escape and pattern[pos] == "\\":
            pos += 2
            continue
        elif pattern[pos] == "{":
            if bracketdepth == 0 and pos > start:
                # print 'literal:', pattern[start:pos]
                items.append([pattern[start:pos]])
                start = pos
            bracketdepth += 1
        elif pattern[pos] == "}":
            bracketdepth -= 1
            if bracketdepth == 0:
                # print 'expression:', pattern[start+1:pos]
                expr = pattern[start + 1 : pos]
                item = parse_expression(expr, escape)
                if item is None:  # not a range or sequence
                    items.extend([["{"], parse_pattern(expr, escape), ["}"]])
                else:
                    items.append(item)
                start = pos + 1  # skip the closing brace
        pos += 1

    if bracketdepth != 0:  # unbalanced braces
        raise UnbalancedBracesError("Unbalanced braces: '%s'" % pattern)

    if start < pos:
        items.append([pattern[start:]])

    return ("".join(item) for item in product(*items))


def parse_expression(expr: str, escape: bool) -> Optional[Iterable[str]]:
    int_range_match = int_range_re.match(expr)
    if int_range_match:
        return make_int_range(*int_range_match.groups())

    char_range_match = char_range_re.match(expr)
    if char_range_match:
        return make_char_range(*char_range_match.groups())

    return parse_sequence(expr, escape)


def parse_sequence(seq: str, escape: bool) -> Optional[Iterator[str]]:
    # sequence -> chain(*sequence_items)
    start = 0
    pos = 0
    bracketdepth = 0
    items: list[Iterable[str]] = []

    # print 'sequence:', seq
    while pos < len(seq):
        if escape and seq[pos] == "\\":
            pos += 2
            continue
        elif seq[pos] == "{":
            bracketdepth += 1
        elif seq[pos] == "}":
            bracketdepth -= 1
        elif seq[pos] == "," and bracketdepth == 0:
            items.append(parse_pattern(seq[start:pos], escape))
            start = pos + 1  # skip the comma
        pos += 1

    if bracketdepth != 0:
        raise UnbalancedBracesError
    if not items:
        return None

    # part after the last comma (may be the empty string)
    items.append(parse_pattern(seq[start:], escape))
    return chain(*items)


def make_int_range(left: str, right: str, incr: Optional[str] = None) -> Iterator[str]:
    if any([s.startswith(("0", "-0")) for s in (left, right) if s not in ("0", "-0")]):
        padding = max(len(left), len(right))
    else:
        padding = 0
    step = (int(incr) or 1) if incr else 1
    start = int(left)
    end = int(right)
    r = range(start, end + 1, step) if start < end else range(start, end - 1, -step)
    fmt = "%0{}d".format(padding)
    return (fmt % i for i in r)


def make_char_range(left: str, right: str, incr: Optional[str] = None) -> str:
    step = (int(incr) or 1) if incr else 1
    start = alphabet.index(left)
    end = alphabet.index(right)
    if start < end:
        return alphabet[start : end + 1 : step]
    else:
        end = end or -len(alphabet)
        return alphabet[start : end - 1 : -step]


if __name__ == "__main__":
    import doctest
    import sys

    failed, _ = doctest.testmod(optionflags=doctest.IGNORE_EXCEPTION_DETAIL)
    if failed:
        sys.exit(1)
