import cosmica


def test_version() -> None:
    version = cosmica.__version__
    assert isinstance(version, str)
    assert version.count(".") == 2
    assert version.split(".")[0].isdigit()
    assert version.split(".")[1].isdigit()
    assert version.split(".")[2].isdigit()
