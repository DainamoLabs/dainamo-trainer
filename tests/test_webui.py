import os

from dainamo import webui


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_rebuilds_only_when_the_source_changes(tmp_path):
    web = str(tmp_path)
    _write(os.path.join(web, "src", "pages", "App.tsx"), "one")
    _write(os.path.join(web, "package.json"), "{}")
    assert not webui.is_current(web)  # never built

    _write(os.path.join(web, "dist", "index.html"), "<html>")
    assert not webui.is_current(web)  # built before the launcher kept a record (an older checkout)
    webui.mark_built(web)
    assert webui.is_current(web)

    os.utime(os.path.join(web, "src", "pages", "App.tsx"))  # a checkout touches files without changing them
    _write(os.path.join(web, "node_modules", "x.js"), "not source")
    assert webui.is_current(web)

    _write(os.path.join(web, "src", "pages", "App.tsx"), "two")  # a pull changed it
    assert not webui.is_current(web)
    webui.mark_built(web)
    _write(os.path.join(web, "src", "pages", "New.tsx"), "")  # or added a file
    assert not webui.is_current(web)
    webui.mark_built(web)
    os.remove(os.path.join(web, "dist", "index.html"))  # the build is gone
    assert not webui.is_current(web)
