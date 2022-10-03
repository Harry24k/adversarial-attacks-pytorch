import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torchattacks


def test_import_version():
    print(torchattacks.__version__)
    assert True
