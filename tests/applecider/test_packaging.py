import applecider


def test_version():
    """Check to see that we can get the package version"""
    assert applecider.__version__ is not None
