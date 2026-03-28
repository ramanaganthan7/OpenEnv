"""Kill any process listening on PORT (default 7860). Safe to run if nothing is there."""
import subprocess
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7860

result = subprocess.run(
    f'netstat -ano | findstr :{PORT}',
    shell=True, capture_output=True, text=True
)

killed = 0
for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 5 and f":{PORT}" in parts[1] and parts[3] == "LISTENING":
        pid = parts[4]
        subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                       capture_output=True)
        print(f"Killed PID {pid} on port {PORT}")
        killed += 1

if killed == 0:
    print(f"Port {PORT} is free.")
