import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import subprocess

# === CONFIGURATION ===
# Replace with your actual executable path
EXECUTABLE_COMMAND = ["python3", "/Users/mercedeslo/Documents/jc/analysis/getFastDataForNextRace.py"]
# Example for script: EXECUTABLE_COMMAND = ["python3", "/path/to/your_script.py"]

# === USER-INPUT HKT TIMES ===
# Format: "YYYY-MM-DD HH:MM"
start_time_hkt_str = "2025-06-14 15:58"
stop_time_hkt_str  = "2025-06-14 23:20"

# === TIMEZONE SETUP ===
hkt = ZoneInfo("Asia/Hong_Kong")
local = datetime.now().astimezone().tzinfo

# === CONVERT TO LOCAL TIMEZONE ===
start_time_local = datetime.strptime(start_time_hkt_str, "%Y-%m-%d %H:%M").replace(tzinfo=hkt).astimezone(local)
stop_time_local  = datetime.strptime(stop_time_hkt_str, "%Y-%m-%d %H:%M").replace(tzinfo=hkt).astimezone(local)

def my_task():
    print(f"[{datetime.now()}] Running task...")
    try:
        subprocess.run(EXECUTABLE_COMMAND, check=True)
        print(f"[{datetime.now()}] Task completed.")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Task failed: {e}")

# === WAIT UNTIL START TIME ===
while datetime.now().astimezone() < start_time_local:
    time_to_wait = (start_time_local - datetime.now().astimezone()).total_seconds()
    print(f"Waiting to start... {round(time_to_wait)}s remaining")
    time.sleep(min(time_to_wait, 60))

# === RUN EVERY 3 MINUTES UNTIL STOP TIME ===
while datetime.now().astimezone() < stop_time_local:
    my_task()
    time.sleep(180)  # 3 minutes

print("All tasks complete. Exiting.")