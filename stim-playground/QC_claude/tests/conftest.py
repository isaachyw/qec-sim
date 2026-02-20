"""
Pytest configuration for QC_claude tests.

Adds the stim-playground directory to sys.path so that
`import QC_claude` works when pytest is run from stim-playground/.
"""
import sys
import os

# QC_claude/tests/ → QC_claude/ → stim-playground/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
