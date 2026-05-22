import pytest

@pytest.mark.dummy
def test_dummy():
    assert True

def main():
    pytest.main(["-m", "dummy"])
    
    test_dummy()

if __name__ == "__main__":
    main()