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

"""Implementation of Prefix Cache

PrefixCache
- maintain LRU in PrefixCache

1. Get longest common prefix from PrefixCacheTrie. Update LRU.
2. If full matched, return cache, done.
3. Save key value into HBMCache
4. If not enough remained size, evict Key in PrefixTrie and HBMCache by LRU
   to a proportion, retry save to HBMCache again.
5. Save the Key into PrefixCacheTrie.
6. Saved cached with shorter one would never be used since longer one replaced.
   The shorter one would be erased first since using LRU.
   Could be better to evict first since we know it will not be used anymore.

PrefixCacheTrie

1. Insert longer Key replace the shorter one. Which shorter one will never
   be used, and got evicted by LRU in the future.
2. Save the idx point to the longest Key.
3. Get longest common prefix return Key with matched length.
   Assume Key is 1 to 1 match to token, which can be used to slice cache Value.
3. Erase the Key traverse the trie. If end at a leaf, remove the node.
   If origin key idx point to the descendant which is removed,
   change the key idx to another leaf.

HBMCache

1. Calculate the used size and maintain map[Key, Value] to get the stored Value.
2. Save return fail if the size is exceed the remained size.
   Calculated at the first.

"""

from typing import Tuple, Any, Optional
import dataclasses


Token = int
# Tuple of tokens from prompt
Key = Tuple[Token, ...]
Prefix = Any  # KVCache for one prompt


@dataclasses.dataclass
class Value:
  """This is the object stored in the hbm and contains the actual KVcache"""

  prefix: Prefix
  true_length: int
  padded_length: int
  tokens: list[int]


class PrefixCacheTrie:
  """Stores prefix tokens as a trie for fast lookup index. Not thread safe."""

  @dataclasses.dataclass
  class Node:
    # We store only a longest Key share the same prefix
    saved_key_idx: int
    next: dict[Token, "PrefixCacheTrie.Node"] = dataclasses.field(default_factory=dict)

    def is_leaf(self):
      return len(self.next) == 0

  def __init__(self):
    self._saved_keys: list[Key] = []
    self._root_dict: dict[Token, PrefixCacheTrie.Node] = {}

  def insert(self, key: Key):
    """Insert key into the trie."""
    if len(key) == 0:
      return

    if key[0] not in self._root_dict:
      new_saved_key_idx = self._append_new_saved_key()
      self._root_dict[key[0]] = PrefixCacheTrie.Node(saved_key_idx=new_saved_key_idx)

    node = self._root_dict[key[0]]
    for token in key[1:]:
      if token not in node.next:
        if node.is_leaf():
          node.next[token] = PrefixCacheTrie.Node(saved_key_idx=node.saved_key_idx)
        else:
          new_saved_key_idx = self._append_new_saved_key()
          node.next[token] = PrefixCacheTrie.Node(saved_key_idx=new_saved_key_idx)

      node = node.next[token]

    if node.is_leaf():
      self._saved_keys[node.saved_key_idx] = key

  def get_longest_common_prefix_key(self, key: Key) -> Optional[Key]:
    """Get the key with longest common prefix.
    If not found at least one token match, return None."""
    if len(key) == 0 or key[0] not in self._root_dict:
      return None

    node = self._root_dict[key[0]]
    for token in key[1:]:
      if token not in node.next:
        break
      node = node.next[token]

    return self._saved_keys[node.saved_key_idx]

  def erase(self, key: Key) -> bool:
    """Erase key in trie. Return False if key is not found."""

  def _append_new_saved_key(self) -> int:
    """Return idx of new append key."""
    idx = len(self._saved_keys)
    self._saved_keys.append(())
    return idx


class HBMCache:
  """Stores kv cache values in HBM and supports eviction using LRU.
  Cache is distributed across all devices on the VM.
  """

  def __init__(self, max_size_bytes: int):
    """
    max_size_bytes: Total amount of HBM to use for cache
    """
    self._max_size_bytes = max_size_bytes

  def add_to_cache(self, key: Key, value: Value):
    pass

  def retrieve_from_cache(self, key: Key) -> Optional[Value]:
    pass


class PrefixCache:
  """Store Prefix KV cache"""

  def __init__(self, hbm_bytes: int):
    """
    hbm_bytes: Total amount of HBM to use for cache. If cache is full,
       evict least-recently used entries (LRU).
    """
    self.hbm_cache = HBMCache(max_size_bytes=hbm_bytes)
    self.trie = PrefixCacheTrie()

  def common_prefix_length(self, key1: Key, key2: Key) -> int:
    """Returns length of longest common prefix"""

  def save(self, key: Key, value: Value) -> bool:
    """Save key/value to the cache"""
    # Evict any rows if cache is full
    # Store to self.hbm_cache
    # Add to trie

  def load(self, key: Key) -> Optional[Value]:
    """Tries to load key from the cache. Returns Value of longest match."""
    # Find key with longest matching prefix in trie
    # fetch value for the key from hbm_cache

  def clear(self):
    """Clear entire cache"""
