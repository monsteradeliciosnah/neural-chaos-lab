def test_systems_smoke():
    try:
        mod = __import__("neural_chaos_lab.systems", fromlist=["*"])
    except Exception:
        mod = __import__("neural_chaos_lab.systems", fromlist=["*"])
    names = [n for n in dir(mod) if not n.startswith("_")]
    assert isinstance(names, list)
