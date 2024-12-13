from bris import cf

def test_empty():
    q, w, e = cf.get_list([])
    assert len(q) == 0
    assert len(w) == 0
    assert len(e) == 0

def test_1():
    q, w, e = cf.get_list(["u_800"])
    assert q == {"pressure": ("pressure", [800])}
    assert w == {"x_wind_pl": "pressure"}
    assert e == {"u_800": ("x_wind_pl", 0)}

def test_2():
    q, w, e = cf.get_list(["u_800", "u_700", "v_700", "2t", "10u"])
    print(q, w, e)
    assert len(q) == 4
