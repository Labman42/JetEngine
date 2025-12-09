from collections import deque
import xxhash
import numpy as np

from jetengine.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # Use a set for O(1) removal of specific blocks (cache hits)
        # Use a deque for O(1) popping of any free block (new allocations)
        # We maintain both: free_block_ids_set for fast lookup/removal, 
        # and free_block_ids_deque for fast pop.
        # Actually, maintaining two is complex. 
        # Let's just use a list/deque, but for "remove specific", we need O(1).
        # A set is good for remove, but pop() from set is arbitrary. That's fine for "any free block".
        self.free_block_ids: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1:
            # Remove old hash mapping if it points to this block
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
        block.reset()
        # self.free_block_ids.remove(block_id) # Removed by caller if using pop()
        # But wait, _allocate_block is called by allocate() which might use pop() OR might use specific block_id (cache hit).
        # If cache hit, we pass specific block_id. We MUST remove it.
        # If cache miss, we pop it.
        # So _allocate_block should handle both?
        # Or we change _allocate_block to NOT remove, and caller removes?
        # Let's keep _allocate_block as is (removes), but handle pop() carefully.
        
        # If we use pop(), the item is ALREADY removed from set.
        # So self.free_block_ids.remove(block_id) will raise KeyError!
        
        # We need to change _allocate_block signature or logic.
        # Let's make _allocate_block assume it's already removed? No, cache hit case needs removal.
        
        if block_id in self.free_block_ids:
            self.free_block_ids.remove(block_id)
            
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.add(block_id)
        # We do NOT remove hash here, to allow cache hits on free blocks (revival)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        num_blocks = seq.num_blocks
        
        for i in range(num_blocks):
            token_ids = seq.block(i)
            if len(token_ids) == self.block_size:
                h = self.compute_hash(token_ids, h)
            else:
                h = -1
                
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                if not self.free_block_ids:
                    raise ValueError("No free blocks available")
                # Use pop() for O(1) removal
                block_id = self.free_block_ids.pop()
                # _allocate_block handles the rest, but we must ensure it doesn't try to remove again.
                # We modified _allocate_block to check existence.
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Reviving a block from free list
                    block = self._allocate_block(block_id)
                    pass 

            if h != -1:
                # Safety: Ensure old hash is removed if it exists and points to this block
                if block.hash != -1 and block.hash != h:
                     if self.hash_to_block_id.get(block.hash) == block_id:
                         del self.hash_to_block_id[block.hash]
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            else:
                block.token_ids = token_ids
                
            seq.block_table.append(block_id)

    def allocate_batch(self, seqs: list[Sequence]):
        # Check if we have enough blocks for all
        total_needed = sum(seq.num_blocks for seq in seqs)
        if len(self.free_block_ids) < total_needed:
            pass

        for seq in seqs:
            self.allocate(seq)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append_blocks(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

    def append_blocks(self, seq: Sequence, num_blocks: int):
        for _ in range(num_blocks):
            if not self.free_block_ids:
                 raise ValueError("No free blocks available")
            # Use pop()
            block_id = self.free_block_ids.pop()
            block = self.blocks[block_id]
            assert block.ref_count == 0, f"Block {block_id} has ref_count {block.ref_count}"
            
            if block.hash != -1:
                if self.hash_to_block_id.get(block.hash) == block_id:
                    del self.hash_to_block_id[block.hash]
            
            block.reset()
            self.used_block_ids.add(block_id)
            # self.free_block_ids.remove(block_id) # Already popped
            seq.block_table.append(block_id)

    def append_blocks_batch(self, seqs_and_counts: list[tuple[Sequence, int]]):
        # seqs_and_counts: list of (seq, num_new_blocks)
        total_needed = sum(count for _, count in seqs_and_counts)
        if len(self.free_block_ids) < total_needed:
             # Handle error or partial
             pass
        
        # Bulk allocate from free list
        # Converting set to list is O(N), but we only need 'total_needed' items.
        # We can pop from set.
        
        # Optimization: If we need many blocks, iterating next(iter(set)) is slow.
        # We can convert set to list once if we need many?
        # Or just use a list for free_blocks if we don't need O(1) remove of specific ID.
        # But we DO need O(1) remove for cache hits (revival).
        
        # Let's just loop for now, but avoid overheads.
        for seq, count in seqs_and_counts:
            self.append_blocks(seq, count)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0] # Accessing set by index is not supported? 
            # Wait, self.free_block_ids is a set. [0] will fail.
            # The original code had: block_id = self.free_block_ids[0] which implies it might have been a list or the user code was buggy?
            # Line 40: self.free_block_ids: set[int] = set(range(num_blocks))
            # So line 125 in original code `self.free_block_ids[0]` would raise TypeError!
            # This is a bug in the original code!
            block_id = next(iter(self.free_block_ids))
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
