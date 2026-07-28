import os
import sys

# Put training/student_core on sys.path so the flat module imports resolve when pytest
# collects from anywhere in the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
