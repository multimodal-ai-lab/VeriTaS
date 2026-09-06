import pytest

from scrapemm.secrets import SECRETS_PATH


@pytest.fixture(autouse=True, scope="function")
def run_before_each_test():
    assert SECRETS_PATH.exists(), "Please set up the secrets first before testing!"
