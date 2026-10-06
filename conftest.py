import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'tests')]   # `swarm`, `api` and the shared fakes in tests/ import cleanly
