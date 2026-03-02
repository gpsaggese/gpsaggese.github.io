pip install pyright

which pyright-langserver

/usr/local/bin/pyright-langserver

Save in .claude/lsp.json

{
  "languageServers": {
    "python": {
      "command": "pyright-langserver",
      "args": ["--stdio"],
      "filetypes": ["py"]
    }
  }
}

cc

/lsp status

Python: running (pyright-langserver)
Indexed: <files>
