Interactive demo & Unity scaffold
=================================

We've added a prototype interactive web demo (FastAPI) and a Unity scaffold
to help explore and port the simulation.

Run the web demo locally (Python):

1. Install dependencies:
   - cd viz/web
   - python -m pip install -r requirements.txt

2. Start the FastAPI backend (serves POST /run):
   - python viz/web/backend.py
   - or: uvicorn viz.web.backend:app --reload --host 0.0.0.0 --port 8000

3. Open the static frontend:
   - Open viz/web/static/index.html in a browser and press Run to send
     requests to the backend. If you serve the static file from a different
     origin, ensure CORS is configured appropriately.

Run with Docker:

- cd viz/web
- docker build -t quantum-demo .
- docker run -p 8000:8000 quantum-demo
- Visit http://localhost:8000/static/index.html (or open the static file)

Unity scaffold:

- See viz/unity/QuantumSim.cs for a simple MonoBehaviour scaffold and notes
  on porting the Python simulation into Unity/C#.
- See viz/unity/README.md for further instructions and porting tips.

----------

NOTE: This section was prepended to the README to document the new demo and
scaffold. If you had additional README content, re-add it below this section
as needed.
