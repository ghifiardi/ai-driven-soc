import jwt


def test_pyjwt_available():
    assert getattr(jwt, "__version__", "")
