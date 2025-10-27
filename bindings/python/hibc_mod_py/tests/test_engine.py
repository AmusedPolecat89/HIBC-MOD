import json, hibc_mod

def test_roundtrip():
    e = hibc_mod.PyDataEngine("my_db")
    cfg = json.loads(e.config_json())
    rs = e.search([0.1]*cfg["vector_dim"], 3)
    assert len(rs) == 3
    assert isinstance(rs[0].id, str)
