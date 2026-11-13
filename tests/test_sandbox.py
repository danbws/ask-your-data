import pandas as pd

from ask_your_data.sandbox import run_python

FRAME = pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})


def test_runs_pandas_code_on_df():
    res = run_python("print(df['b'].sum()); print(len(df))", FRAME)
    assert res.ok
    assert "60.0" in res.stdout and "3" in res.stdout


def test_error_is_reported_not_raised():
    res = run_python("print(df['missing'])", FRAME)
    assert not res.ok
    assert "KeyError" in res.stderr
    assert res.to_text().startswith("ERROR")


def test_timeout_kills_infinite_loop():
    res = run_python("while True:\n    pass", FRAME, timeout_s=1.5)
    assert res.timed_out and not res.ok


def test_output_is_truncated():
    res = run_python("print('x' * 100000)", FRAME)
    assert res.ok and len(res.stdout) < 7000 and "truncated" in res.stdout


def test_network_is_blocked():
    code = "import socket\ns = socket.socket()\ns.connect(('127.0.0.1', 9))\n"
    res = run_python(code, FRAME)
    assert not res.ok and "blocked by sandbox" in res.stderr


def test_write_outside_temp_dir_is_blocked(tmp_path):
    target = tmp_path / "evil.txt"
    res = run_python(f"open({str(target)!r}, 'w').write('x')", FRAME)
    assert not res.ok and "blocked by sandbox" in res.stderr
    assert not target.exists()


def test_write_inside_temp_dir_is_allowed():
    res = run_python("open('scratch.txt', 'w').write('ok'); print(open('scratch.txt').read())",
                     FRAME)
    assert res.ok and "ok" in res.stdout


def test_subprocess_is_blocked():
    res = run_python("import subprocess\nsubprocess.run(['cmd', '/c', 'echo', 'hi'])", FRAME)
    assert not res.ok and "blocked by sandbox" in res.stderr
