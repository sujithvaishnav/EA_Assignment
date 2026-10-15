"""
run_all.py
----------
Runs both the FastAPI Backend (Port 8000) and Streamlit Dashboard (Port 8501)
concurrently from a single terminal command.
Press Ctrl+C to shut down both servers cleanly.

Usage:
    & "d:\\EA_Assignment\\venv\\Scripts\\python.exe" run_all.py
"""

import os
import sys
import subprocess
import time
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
VENV_STREAMLIT = os.path.join(BASE_DIR, "venv", "Scripts", "streamlit.exe")

if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

print("=" * 70)
print("  STARTING SELF-CORRECTIVE RAG APPLICATION")
print("  FastAPI Backend   : http://localhost:8000 (Swagger: /docs)")
print("  Streamlit Frontend: http://localhost:8501")
print("=" * 70)

# 1. Start FastAPI Backend (Uvicorn)
api_cmd = [
    VENV_PYTHON,
    "-m",
    "uvicorn",
    "api.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]
print("\n[Runner] Launching FastAPI Backend on port 8000...")
api_process = subprocess.Popen(api_cmd, cwd=BASE_DIR)

# Give FastAPI 2 seconds to bind its port
time.sleep(2)

# 2. Start Streamlit Frontend
streamlit_cmd = [
    VENV_STREAMLIT if os.path.exists(VENV_STREAMLIT) else "streamlit",
    "run",
    "streamlit_app.py",
    "--server.port",
    "8501",
    "--server.headless",
    "true",
]
print("[Runner] Launching Streamlit UI on port 8501...\n")
streamlit_process = subprocess.Popen(streamlit_cmd, cwd=BASE_DIR)

print("\n" + "=" * 70)
print("  Both services are running!")
print("  - Backend API: http://localhost:8000/docs")
print("  - Web UI:      http://localhost:8501")
print("  Press Ctrl+C in this terminal to stop both services.")
print("=" * 70 + "\n")


def shutdown(sig, frame):
    print("\n[Runner] Shutting down services...")
    try:
        api_process.terminate()
        streamlit_process.terminate()
        api_process.wait()
        streamlit_process.wait()
    except Exception:
        pass
    print("[Runner] All services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

try:
    while True:
        if api_process.poll() is not None:
            print("[Runner] Warning: FastAPI process exited.")
            break
        if streamlit_process.poll() is not None:
            print("[Runner] Warning: Streamlit process exited.")
            break
        time.sleep(1)
except KeyboardInterrupt:
    shutdown(None, None)
