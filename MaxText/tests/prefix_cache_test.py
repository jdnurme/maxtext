# Copyright 2025 Google LLC
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

"""Prefix Cache Test"""

from prefix_cache import PrefixCacheTrie

import unittest


class PrefixCacheTrieTest(unittest.TestCase):
  """Test for PrefixCacheTrie."""

  def test_get_longest_common_prefix_key(self):
    trie = PrefixCacheTrie()
    key = (1, 2, 3, 4)
    trie.insert(key)
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4, 5)) == key
    assert trie.get_longest_common_prefix_key((1, 2, 6, 7)) == key
    assert trie.get_longest_common_prefix_key((2, 3, 4, 5)) is None

  def test_insert_longer_key_replace_shorter_key(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 3, 4))
    assert trie.get_longest_common_prefix_key((1, 2, 3)) == (1, 2, 3, 4)

  def test_insert_key_with_different_suffix_will_store_another_one(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3, 4))
    trie.insert((1, 2, 3, 5))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert trie.get_longest_common_prefix_key((1, 2, 3, 5)) == (1, 2, 3, 5)

  def test_insert_shorter_key_will_not_replace_longer_key(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3, 4))
    trie.insert((1, 2, 3))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4)) == (1, 2, 3, 4)

  def test_insert_multiple_key_and_get_longest_common_prefix(self):
    trie = PrefixCacheTrie()
    trie.insert((1, 2, 3))
    trie.insert((1, 2, 4))
    trie.insert((11, 2, 3))
    trie.insert((11, 2))
    trie.insert((11, 3, 4))
    assert trie.get_longest_common_prefix_key((1, 2, 3, 4, 5))[:3] == (1, 2, 3)
    assert trie.get_longest_common_prefix_key((1, 2, 4, 5, 6))[:3] == (1, 2, 4)
    assert trie.get_longest_common_prefix_key((11, 2, 3, 4))[:3] == (11, 2, 3)
    assert trie.get_longest_common_prefix_key((11, 3, 6))[:2] == (11, 3)


if __name__ == "__main__":
  unittest.main()
