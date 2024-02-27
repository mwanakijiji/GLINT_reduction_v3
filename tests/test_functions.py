import sys, os

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, target_dir)

from modules import *
from modules import functions

def test_the_fcn():
    # Test the_fcn function
    functions.the_fcn()  # Call the function
    # Add assertions to check the expected output
    # For example:
    assert 1<2
