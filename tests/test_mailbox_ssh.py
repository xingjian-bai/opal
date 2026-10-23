"""The broker's ssh path, with a fake ssh on PATH."""

import os
import subprocess
import sys
import textwrap
import threading

from opal.backends.mailbox import _Folder

FAKE_SSH = textwrap.dedent(
    """\
    #!{py}
    import os, select, subprocess, sys
    got = b""
    if select.select([0], [], [], 0.5)[0]:
        got = os.read(0, 1000)
    open({log!r}, "ab").write(repr(got).encode() + b"\\n")
    sys.stderr.write("Warning: Permanently added 'pod' (ED25519) to the list of known hosts.\\n")
    sys.exit(subprocess.run(["sh", "-c", sys.argv[-1]]).returncode)  # the remote command, run here
    """
)


def fake_ssh(tmp_path):
    b, log = tmp_path / "bin", tmp_path / "stdin.log"
    b.mkdir()
    (b / "ssh").write_text(FAKE_SSH.format(py=sys.executable, log=str(log)))
    (b / "ssh").chmod(0o755)
    return b, log


def test_missing_file_reads_as_none_although_ssh_writes_to_stderr(tmp_path, monkeypatch):
    b, _ = fake_ssh(tmp_path)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    box = tmp_path / "box"
    box.mkdir()
    (box / "present.json").write_bytes(b"{}")
    f = _Folder(str(box), ssh="gpu-host")
    assert f.read("present.json") == b"{}"
    assert f.read("missing.json") is None


def test_ssh_reads_do_not_take_the_brokers_stdin(tmp_path):
    b, log = fake_ssh(tmp_path)
    code = f"from opal.backends.mailbox import _Folder; print(_Folder({str(tmp_path)!r}, ssh='h').read('x.json'))"
    env = {**os.environ, "PATH": f"{b}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-c", code],
        input=b"typed into the broker's terminal\n",
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert r.returncode == 0 and r.stdout.strip() == b"None", r
    assert log.read_text().strip() == "b''"


READS_FAIL = textwrap.dedent(
    """\
    #!/bin/sh
    case "$*" in
      *"test -f"*) echo "Permission denied (publickey)." >&2; exit 255 ;;
    esac
    for last; do :; done
    exec sh -c "$last"
    """
)


def test_broker_reports_repeated_failures_and_can_give_up(tmp_path, monkeypatch, capsys):
    import pytest

    from opal.backends import mailbox
    from opal.backends.base import KeepBackend

    b = tmp_path / "bin"
    b.mkdir()
    (b / "ssh").write_text(READS_FAIL)
    (b / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(mailbox, "RETRY_SECONDS", 0.05)
    (tmp_path / "box" / "run1").mkdir(parents=True)
    lines = []
    with pytest.raises(SystemExit, match="giving up"):
        mailbox.run_broker(
            str(tmp_path / "box"), KeepBackend(), tmp_path / "work", ssh="h", run="run1", log=lines.append,
            give_up_after=0.5,
        )  # fmt: skip
    assert any("failed reads of the mailbox in a row" in m and "Permission denied" in m for m in lines)


def test_ssh_listing_failures_are_not_treated_as_an_empty_mailbox(tmp_path, monkeypatch):
    import pytest

    b = tmp_path / "bin"
    b.mkdir()
    (b / "ssh").write_text(READS_FAIL)
    (b / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(RuntimeError, match="Permission denied"):
        _Folder(str(tmp_path), ssh="h").runs()


def test_broker_can_give_up_before_it_selects_a_run(tmp_path, monkeypatch):
    import pytest
    from opal.backends import mailbox
    from opal.backends.base import KeepBackend

    clock = [100.0]
    owner = threading.get_ident()
    real_time, real_sleep = mailbox.time.time, mailbox.time.sleep

    def sleep(seconds):
        if threading.get_ident() != owner:
            return real_sleep(seconds)
        clock[0] += seconds
        assert clock[0] < 102.0, "initial mailbox selection ignored give_up_after"

    def fail(*args):
        raise OSError("mailbox unreadable")

    monkeypatch.setattr(mailbox.time, "time", lambda: clock[0] if threading.get_ident() == owner else real_time())
    monkeypatch.setattr(mailbox.time, "sleep", sleep)
    monkeypatch.setattr(mailbox, "RETRY_SECONDS", 0.1)
    monkeypatch.setattr(mailbox, "_pick_run", fail)
    with pytest.raises(SystemExit, match="giving up"):
        mailbox._serve(_Folder(str(tmp_path)), KeepBackend(), tmp_path, None, 0.1, None, lambda _: None,
                       {}, False, give_up_after=0.5)


def test_remote_missing_mailbox_is_distinct_from_an_empty_one(tmp_path, monkeypatch):
    import pytest

    b, _ = fake_ssh(tmp_path)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _Folder(str(empty), ssh="h").runs() == []
    with pytest.raises(RuntimeError, match="listing failed"):
        _Folder(str(tmp_path / "missing"), ssh="h").runs()


ZSH_SSH = textwrap.dedent(
    """\
    #!/bin/sh
    for last; do :; done
    exec zsh -c "$last"
    """
)


def test_ssh_listing_works_when_the_remote_login_shell_is_zsh(tmp_path, monkeypatch):
    import shutil

    import pytest

    if not shutil.which("zsh"):
        pytest.skip("zsh is not installed")
    b = tmp_path / "bin"
    b.mkdir()
    (b / "ssh").write_text(ZSH_SSH)  # the remote command runs in zsh, as on hosts whose login shell is zsh
    (b / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(tmp_path))
    box = tmp_path / "box"
    box.mkdir()
    f = _Folder(str(box), ssh="h")
    assert f.runs() == []  # zsh fails on a glob that matches nothing: "no matches found: */"
    (box / "run1").mkdir()
    (box / "run1" / "RUN").write_text("{}")
    assert f.runs() == ["run1"]
