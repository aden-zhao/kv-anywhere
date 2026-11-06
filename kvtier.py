"""vLLM's file-system KV tier, plus a record of which blocks get read.

vLLM's tier never deletes anything, so tier_gc.py keeps it under budget. To evict well,
tier_gc.py needs to know which blocks are still being reused, but tmpfs doesn't update a
file's access time on read. This subclass stamps it whenever vLLM loads a block:
    atime = last read, mtime = when written.
Everything else (hashing, file layout, GPU <-> CPU <-> file copies) is vLLM's tier.

Use it as an out-of-tree tier (kvtier.py must be importable by the vLLM server):
    "secondary_tiers": [{"type": "ReadStampedTier", "module_path": "kvtier", "root_dir": ...}]
"""
import os
import time

from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager


class ReadStampedTier(FileSystemTierManager):
    def submit_load(self, job_metadata):
        now = time.time()
        for key in job_metadata.keys:
            path = self.file_mapper.get_file_name(key)
            try:
                os.utime(path, (now, os.stat(path).st_mtime))
            except FileNotFoundError:
                pass  # tier_gc.py deleted it since the lookup; vLLM recomputes the block
        super().submit_load(job_metadata)
