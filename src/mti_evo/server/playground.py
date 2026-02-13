"""
MTI-EVO Playground Manager
==========================
Manages execution of python scripts in the 'playground' directory.
Handles process spawning, log capturing, and auditing.
"""
import os
import time
import json
import sys
import subprocess

class PlaygroundManager:
    """
    Manages execution of playground scripts.
    Handles auditing (file logging) and process management.
    """
    def __init__(self, playground_dir):
        self.playground_dir = playground_dir
        self.running_processes = {} # {pid: {'proc': proc, 'log_file': path, 'script': name}}
        self.logs_dir = os.path.join(playground_dir, "..", "logs", "playground")
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def list_scripts(self):
        """Returns list of scripts with metadata."""
        try:
            # Read Manifest
            manifest = {}
            manifest_path = os.path.join(self.playground_dir, "manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                except:
                    pass

            scripts = []
            if os.path.exists(self.playground_dir):
                for f in os.listdir(self.playground_dir):
                    if f.endswith(".py") and not f.startswith("__"):
                        meta = manifest.get(f, {
                            "name": f, 
                            "description": "No description provided.",
                            "args": []
                        })
                        # Ensure filename is attached
                        meta['filename'] = f
                        scripts.append(meta)
            
            # Sort by Name
            return sorted(scripts, key=lambda x: x['name'])
        except Exception as e:
            return []

    def run_script(self, script_name, args=None):
        """Spawns a script execution with optional arguments."""
        # Validate existence (simple check)
        if not os.path.exists(os.path.join(self.playground_dir, script_name)):
             return {"error": "Script not found"}
            
        script_path = os.path.join(self.playground_dir, script_name)
        timestamp = int(time.time())
        log_filename = f"{timestamp}_{script_name}.log"
        log_path = os.path.join(self.logs_dir, log_filename)
        
        try:
            # Open log file for redirection
            log_file = open(log_path, "w")
            
            # Build Command
            cmd = [sys.executable, "-u", script_path]
            if args and isinstance(args, list):
                cmd.extend(args)
            
            # Spawn subprocess (Non-blocking)
            proc = subprocess.Popen(
                cmd,
                cwd=self.playground_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            self.running_processes[proc.pid] = {
                'proc': proc,
                'log_file_handle': log_file,
                'log_path': log_path,
                'script': script_name,
                'start_time': timestamp
            }
            
            return {"pid": proc.pid, "status": "started", "log_file": log_filename}
        except Exception as e:
            return {"error": str(e)}

    def get_log_tail(self, pid, lines=50):
        """Reads the last N lines of the log file for a process."""
        try:
            pid = int(pid)
        except:
            return {"error": "Invalid PID"}

        if pid not in self.running_processes:
             return {"error": "Process not active (or finished and cleared)"}
             
        info = self.running_processes[pid]
        proc = info['proc']
        
        # Check if done
        status = "running"
        if proc.poll() is not None:
            status = f"finished (code {proc.returncode})"
            # Close handle if not closed
            if not info['log_file_handle'].closed:
                 info['log_file_handle'].close()

        # Read file from disk
        try:
            if os.path.exists(info['log_path']):
                with open(info['log_path'], 'r') as f:
                    content = f.readlines()
                    return {
                        "pid": pid,
                        "status": status,
                        "lines": content[-lines:],
                        "full_path": info['log_path']
                    }
            return {"pid": pid, "status": status, "lines": [], "error": "Log file not found"}
        except Exception as e:
            return {"error": f"Failed to read log: {e}"}

    def stop_process(self, pid):
         try:
             pid = int(pid)
             if pid in self.running_processes:
                 proc = self.running_processes[pid]['proc']
                 proc.kill()
                 return {"status": "killed"}
             return {"error": "Process not found"}
         except:
             return {"error": "Invalid PID"}
