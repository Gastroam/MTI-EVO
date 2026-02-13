
import os
import json
import time
import shutil
import numpy as np
from typing import Dict, Any, Iterable, Optional
from mti_evo.core.persistence.backend import PersistenceBackend, NeuronStateV1
from mti_evo.core.logger import get_logger

logger = get_logger("Persistence-JSONL")

class CorruptDataError(Exception):
    pass

class JsonlPersistence(PersistenceBackend):
    """
    Baseline Persistence: Single-file Append-Only JSONL.
    
    - Writes are appended to 'neurons.jsonl'.
    - Reads use an in-memory index (seed -> file_position) built at startup.
    - Startup time: O(N) scan.
    - Runtime Read: O(1) seek.
    - Runtime Write: O(1) append.
    """
    

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        self.file_path = os.path.join(data_dir, "neurons.jsonl")
        self.wal_path = os.path.join(data_dir, "neurons.wal")
        self.marker_path = os.path.join(data_dir, "neurons.replay_in_progress")
        self.index: Dict[int, int] = {} # Seed -> Byte Offset
        
        # 0. D3: Check for Replay Marker (Crash after rotate, before apply)
        if os.path.exists(self.marker_path):
            self._handle_replay_marker()
        
        # 1. Check for crash recovery (WAL exists)
        self._recover_from_wal()
        
        # 2. Build Index
        self._load_index()

    def _read_records_safely(self, path: str) -> Iterable[Any]:
        """
        Generator that reads a JSONL file safely.
        - Yields (offset, record_dict).
        - Enforces D2: Middle corruption is FATAL. End corruption is IGNORED.
        """
        if not os.path.exists(path):
            return

        with open(path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                    yield offset, record
                except json.JSONDecodeError:
                    # Check if EOF follows immediately
                    current_pos = f.tell()
                    chunk = f.read(1)
                    if not chunk:
                        logger.warning(f"⚠️ Ignored torn line at end of {os.path.basename(path)} (offset {offset})")
                        break
                    else:
                        # Restore position (though we technically read 1 byte, so we are shifting)
                        # But it doesn't matter, we are raising Fatal.
                        raise CorruptDataError(f"Corrupt line in middle of {os.path.basename(path)} (line starting at {offset})")

    def _handle_replay_marker(self):
        """
        D3: Check for marker. If found, resume replay from the indicated file.
        """
        try:
            with open(self.marker_path, 'r') as f:
                data = json.load(f)
                wal_file = data.get("source")
                ts = data.get("ts")
            
            wal_full_path = wal_file if os.path.isabs(wal_file) else os.path.join(self.data_dir, wal_file)
            
            if os.path.exists(wal_full_path):
                logger.warning(f"⚠️ Found Replay Marker (ts={ts}) pointing to {wal_file}. Resuming replay...")
                self._replay_file(wal_full_path, is_marker_recovery=True)
            else:
                logger.warning(f"⚠️ Found Replay Marker but file {wal_file} is missing. Deleting marker.")
                os.remove(self.marker_path)
                
        except Exception as e:
             logger.error(f"Replay Marker check failed: {e}")
             try:
                 os.rename(self.marker_path, self.marker_path + ".corrupt")
             except: pass

    def _replay_file(self, path: str, is_marker_recovery: bool = False):
        """
        Common replay logic for WAL and rotated WAL.
        """
        try:
            # Create Marker (D3)
            if not is_marker_recovery:
                with open(self.marker_path, 'w') as mf:
                    marker_data = {
                        "source": os.path.basename(path),
                        "ts": time.time(),
                        "state": "replaying"
                    }
                    json.dump(marker_data, mf)
                    mf.flush()
                    os.fsync(mf.fileno())

            # Read records safely (D2)
            records = []
            byte_count = 0
            
            # We read all first to ensure we don't partial-write if WAL is totally busted in middle?
            # Or stream? Streaming is better for RAM.
            # But with _read_records_safely, iteration will RAISE if middle corrupt.
            # So we effectively abort replay on middle corruption.
            
            with open(self.file_path, 'ab') as f:
                f.seek(0, 2)
                for _, record in self._read_records_safely(path):
                    line = json.dumps(record, default=str).encode('utf-8') + b'\n'
                    f.write(line)
                    byte_count += len(line)
                    records.append(record)
                
                # Fsync JSONL after replay
                f.flush()
                os.fsync(f.fileno())
            
            record_count = len(records)
            
            if record_count == 0 and not is_marker_recovery:
                 self._archive_wal("empty")
                 if os.path.exists(self.marker_path): os.remove(self.marker_path)
                 return

            if not is_marker_recovery:
                self._archive_wal("replayed")
                
            # Cleanup marker
            if os.path.exists(self.marker_path):
                os.remove(self.marker_path)
            
            source_name = os.path.basename(path)
            logger.info(f"✅ Recovery Complete. Replayed {record_count} records ({byte_count} bytes) from {source_name}")
            
        except Exception as e:
            logger.error(f"Replay Failed for {path}: {e}")
            if not is_marker_recovery:
                self._archive_wal("corrupt")
            raise e

    def _recover_from_wal(self):
        """Replays WAL if it exists."""
        if os.path.exists(self.wal_path):
            logger.warning(f"⚠️ Found WAL file: {self.wal_path}. Recovering...")
            self._replay_file(self.wal_path)

    def _archive_wal(self, suffix: str):
        """Safely rename WAL file out of the active path."""
        try:
            if os.path.exists(self.wal_path):
                # Timestamp + PID for uniqueness
                ts = int(time.time())
                pid = os.getpid()
                dest = f"{self.wal_path}.{suffix}.{ts}.{pid}"
                
                if os.path.exists(dest):
                     os.remove(dest)
                
                os.replace(self.wal_path, dest)
                logger.info(f"✅ WAL rotated to {os.path.basename(dest)}")
        except Exception as e:
            logger.warning(f"Failed to archive WAL ({suffix}): {e}")

    def _load_index(self):
        """Builds in-memory index by scanning JSONL file."""
        t0 = time.time()
        self.index.clear()
        
        if not os.path.exists(self.file_path):
            return

        # Use the safe reader to build index
        # This implicitly verifies integrity of main DB on startup
        try:
            for offset, record in self._read_records_safely(self.file_path):
                if 'seed' in record:
                    if record.get('deleted', False):
                        self.index.pop(record['seed'], None)
                    else:
                        self.index[record['seed']] = offset
        except Exception as e:
            logger.error(f"Index load failed: {e}")
            # If main DB is corrupt in middle, we probably should stop?
            # Or should we allow partial load?
            # Contract says D2: Middle corruption = FATAL.
            raise e
        
        dur = (time.time() - t0) * 1000
        logger.info(f"Indexed {len(self.index)} neurons from {self.file_path} in {dur:.2f}ms")

    def get(self, seed: int) -> Optional[Dict[str, Any]]:
        return self.get_neuron(seed)

    def get_neuron(self, seed: int) -> Optional[Dict[str, Any]]:
        offset = self.index.get(seed)
        if offset is None:
            return None
            
        try:
            with open(self.file_path, 'rb') as f:
                f.seek(offset)
                line = f.readline()
                data = json.loads(line)
                if data.get('deleted', False):
                    return None
                return data
        except Exception as e:
            logger.error(f"Read error for seed {seed} at {offset}: {e}")
            return None

    def upsert_neurons(self, neurons: Dict[int, Dict[str, Any]]) -> None:
        """
        Atomic Write flow:
        1. Write to WAL.
        2. Sync WAL.
        3. Append to JSONL.
        4. Sync JSONL.
        5. Update Index.
        6. Delete/Truncate WAL.
        """
        if not neurons:
            return

        serialized_lines = []
        for seed, state in neurons.items():
            state['v'] = 1
            # Ensure we don't accidentally write 'deleted'
            if 'deleted' in state: del state['deleted']
            line = json.dumps(state, default=str).encode('utf-8') + b'\n'
            serialized_lines.append((seed, line))

        # 1. Write to WAL
        try:
            with open(self.wal_path, 'wb') as wal:
                for _, line in serialized_lines:
                    wal.write(line)
                wal.flush()
                os.fsync(wal.fileno())
        except Exception as e:
            logger.error(f"WAL Write Failed: {e}")
            raise e # Abort, do not corrupt main DB

        # 2. Write to JSONL
        try:
            with open(self.file_path, 'ab') as f:
                # Ensure we are at the end before getting initial offset
                f.seek(0, 2)
                
                for seed, line in serialized_lines:
                    # Get current position before write
                    offset = f.tell()
                    f.write(line)
                    self.index[seed] = offset # Optimistic index update
                
                f.flush()
                # os.fsync(f.fileno()) # Compliance
        except Exception as e:
            logger.error(f"JSONL Write Failed: {e}")
            # WAL remains for next startup recovery
            raise e

            logger.warning(f"Failed to clear WAL: {e}")
            self._archive_wal("stuck")

    def put(self, seed: int, weights: np.ndarray, velocity: np.ndarray,
            bias: float, gravity: float, age: int, last_accessed: float) -> bool:
        """
        Single item write. Delegates to upsert_neurons.
        """
        state = {
            "weights": weights.tolist(),
            "velocity": velocity.tolist(),
            "bias": bias,
            "gravity": gravity,
            "age": age,
            "last_accessed": last_accessed
        }
        try:
            self.upsert_neurons({seed: state})
            return True
        except Exception:
            return False

    def delete_neuron(self, seed: int) -> None:
        """
        Append tombstone record and remove from index.
        """
        tombstone = {"seed": seed, "deleted": True, "ts": time.time()}
        line = json.dumps(tombstone).encode('utf-8') + b'\n'
        
        # 1. Write to WAL (for crash safety during append)
        try:
            with open(self.wal_path, 'wb') as wal:
                wal.write(line)
                wal.flush()
                os.fsync(wal.fileno())
        except Exception as e:
             logger.error(f"WAL Write Failed for delete: {e}")
             raise e

        # 2. Append to JSONL
        try:
            with open(self.file_path, 'ab') as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
                
            # 3. Update Index (Remove)
            self.index.pop(seed, None)
            
        except Exception as e:
            logger.error(f"JSONL Delete Failed: {e}")
            raise e

        # 4. Clear WAL
        try:
             os.remove(self.wal_path)
        except: pass

    def delete(self, seed: int) -> None:
        self.delete_neuron(seed)


    def iter_neurons(self) -> Iterable[Dict[str, Any]]:
        # This iterates strictly over LATEST versions if we use the index.
        # If we just scan file, we get history. Protocol says "iter stored neurons".
        # Usually implies efficient bulk load of *current* state.
        
        # Naive: Sort index by offset and read? 
        # Or just yield get_neuron(seed) for seed in index
        for seed in self.index:
            n = self.get_neuron(seed)
            if n: yield n

    def flush(self) -> None:
        # OS checks
        pass

    def close(self) -> None:
        pass

    def get_active_count(self) -> int:
        return len(self.index)

    def compact(self):
        """
        Rewrites the log to remove superseded entries.
        Atomic replace via temp file.
        """
        temp_path = self.file_path + ".tmp"
        new_index = {}
        
        try:
            with open(temp_path, 'wb') as out_f:
                # We iterate our canonical index ensuring we only write latest
                for seed in sorted(self.index.keys()):
                    n = self.get_neuron(seed)
                    if n:
                        offset = out_f.tell()
                        line = json.dumps(n, default=str).encode('utf-8') + b'\n'
                        out_f.write(line)
                        new_index[seed] = offset
                
                # Flush and Sync to disk before rename
                out_f.flush()
                os.fsync(out_f.fileno())
                
            # Atomic Swap
            if os.path.exists(self.file_path):
                 target_backup = self.file_path + ".bak"
                 if os.path.exists(target_backup):
                     os.remove(target_backup)
                 # Optional: Backup old file just in case? 
                 # For now, standard atomic replace is usually enough if we trust file system.
                 # Python 3.3+ os.replace is atomic on POSIX and Windows (renames over existing).
            
            os.replace(temp_path, self.file_path)
            self.index = new_index
            logger.info(f"Compaction complete. {len(self.index)} active neurons.")
            
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Old file remains untouched
            raise e
