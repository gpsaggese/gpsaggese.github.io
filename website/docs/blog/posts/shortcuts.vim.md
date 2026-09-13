---
title: "Vim Shortcuts"
draft: false
authors:
  - gpsaggese
date: 2026-08-20
categories:
  - Productivity
---

TL;DR: A cheat sheet of Vim commands and concepts.

<!-- more -->

# Command line

- Open and run a command in vim: `> vim -c '...'`
- Print the path of VIMRUNTIME: `> vim --cmd 'echo $VIMRUNTIME' --cmd 'quit'`
- Regenerate the tags (?): `> vim -c "helptags ~/.vim/doc"`
- Skip reading the config `.vimrc`: `> vim -u 'NONE'`

# Brief concepts

**Buffer**
- An in-memory representation of a file's content, once it's loaded into Vim
- Multiple buffers can be open at once (one per file you've edited/loaded)
  - Lists open buffers: `:buffers` 
  - Switches to one buffer: `:b <name/number>` 

**Window (aka viewport)**
- A viewport into a buffer: the visible frame showing (part of) a buffer's content
- Multiple windows can display the same buffer simultaneously
  - E.g., two views scrolled to different lines
- Created by splitting the screen
  - Navigate with `Ctrl-w` + direction (`h/j/k/l`)
  - Closed with `:q` (doesn't close the buffer, just the view)

**Split**
- The act of dividing the screen into multiple windows
- Horizontal split: `:split` or `:sp` (stacks windows top/bottom)
- Vertical split: `:vsplit` or `:vsp` (side by side)
  - Each split is a window
  - Each window shows a buffer

**How they relate**
- Tabs (`:tabnew`) contain layouts of windows
- Windows contain/display buffers
- Buffers hold the actual file content
- In brief: one buffer -> can be shown in many windows -> windows are arranged via
  splits -> splits live inside tabs

# Help

- Get help: `:help TOPIC`
  - `CTRL-d` or `tab` to auto-complete
- Get help on a shortcut: `:help CTRL-W_CTRL-W`
- Open help about TOPIC in a vertical split: `:vert help TOPIC`
  - E.g., `:vert help marks`
- Get help about registers: `:help registers`
- Get help about copying and moving: `:help copy-move`

# Custom shortcuts (from `.vimrc`)

## Leader key
- Leader key is set to comma (default is `\`): `,`

## Editing and movement
- Delete line into the black hole register (doesn't clobber clipboard): `<C-c><C-d>`
- Open file under cursor in a new tab (overrides default "go to declaration"): `gd`
- Shift-Down/Shift-Up pass through as normal `<Down>`/`<Up>` (macOS compatibility)

## Tabs
- Go to next tab: `<C-p>`
- Go to previous tab: `<C-o>`

## Windows
- `<C-w><C-w>` is disabled; use directional `<C-w>h/j/k/l` instead

## Navigating
- Highlight from the current `*` line to the line before the next `*`: `,y`
- Move to the next `*` line: `,n`
- Move to the previous `*` line: `,p`

## Autocorrect
- Autocorrect the word under/before the cursor: `<C-l><C-l>`

## Taglist / tags
- Toggle the tag list panel: `<C-Y><C-Y>`
- Open tag under cursor in a new tab (`:TagTab`): `<C-Y><C-T>`
- Open tag under cursor in a preview window (`:TagPreview`): `<C-Y><C-P>`

## Function keys
- Quit without saving all buffers: `<F1>`
- Toggle search highlighting: `<F2>`
- Toggle spell checking: `<F3>`
- Toggle paste mode: `<F4>`
- Re-sync syntax highlighting from the start of the file: `<F5>`

### Quickfix / buffers (non-diff mode)
- Previous / next quickfix error: `<F9>` / `<F10>`
- Previous / next buffer: `<S-F9>` / `<S-F10>`

### Diff mode (`vimdiff`)
- Refresh the diff view: `<F8>`
- Get diff (pull change from the other file into this one): `<F6>`
- Put diff (push change from this file into the other one): `<F7>`
- Accept left side (copy left file to right, keep it as baseline): `<F9>`
- Accept right side (copy right file to left, overwriting it): `<F10>`

# Settings

- Show value of a variable: `:set var?`
- Disable an option: `:set noXXXX`
- Set for all buffer: `:set XXXX`
- Toggle for all buffer: `:set XXXX!`

- Set only for the current buffer: `:setlocal XXXX`

## Useful settings

- Disable flashing on error: `:set visualbells`
- Disable beeping on error: `:set noerrorbells`
- Set case sensitive/insensitive: `:set ic, set noic`
- Set preview height: `:set previewheight=15`
- Show line number: `:set number` (or `:set nu`)
- To avoid automatic indentation: `:set paste`

## Misc settings

- Show cursor line (using underline): `:set cursorline`
- To highlight the position of the cursor.: `:highlight CursorLine guibg=lightblue ctermbg=lightgray`
- Show all the color scheme: `:highlight`
- Show the palette: `:runtime syntax/colortest.vim`

## Setting path

- Print path: `:set path`
- Look for certain file in path: `:find file_name`
- Update path: `:set path+=/path/`

## Opening files under the cursor

- Open file under the cursor: `gf`
- Jump to a file and line (like in a cfile): `gF`
- Open an URL under the cursor with the default browse: `gx`
- Open file under the cursor in a split window: `CTRL-W, CTRL-F`

## Custom clipboard shortcuts (from `.vimrc`)

- Copy current line to system clipboard (normal or visual mode): `ycc` (not standard)
- Copy `file:line 'content'` reference of line under cursor to clipboard: `ycl`

## Edit command and search history

- Edit search history In command mode: `q:`
- Search in search history: `q/`, `q?`

# Moving

- Move one character at the time: `ijkl`

- Move forward to the start/end of a word: `w, e`
- Move backward to the start/end of a word: `b, ge`

- Go to the start of the file: `gg`
- Go to the end of the file: `G`

- Move to the Home, Middle, or Last of the current screen: `H, M, L`
- Move to the next/previous paragraph: `), (`
- Move to the beginning/end of a line: `0, $`
- Move to the first character of a line: `^`

## Move to char
- Move to the next, previous occurrence of character `x`: `fx, Fx`
- Repeat the last `fx` command: `;`
- Like `fx`, but moves to the character immediately before: `tx`

## Moving on wrapped-around lines
- Move up and down: `gj,gk`
- Move at the beginning/end of a line: `g$,g0`
- Move to the center: `gm`

## Highlighting stuff
- Reselect a visual area: `gv`
- Highlight a word without moving: `*#`

- Block highlight: `CTRL-v`

- Move to the next matching word under the cursor: `*`
- Move to the previous matching word under the cursor: `#`

## Moving the edit window
- Move line with cursor to the center: `zz`
- Move line with cursor to the top of the page: `z<CR>`
- Move line with cursor to the bottom of the page: `z-`

# Editing

- Go to insertion mode: `i`
- Insert at the beginning of the line: `I`
- Insert at the end of the line: `A` (append)
- Replace one char: `r`
- Replace more than one chars: `R`

- Change case of a char: `~`
- Switch case of {motion} text: `g~{motion}`

## Go to normal mode for one command
- Execute one command while in insert mode: `CTRL-O` + command
- Delete the rest of the line while in command mode: `CTRL-O d$`
- Find the next `k` while in command mode: `CTRL-O fk`

- Yank and insert: `c{motion}` yank the text and then insert

## Joining
- Join selected lines (with spaces): `J`
- Join selected lines without spaces: `gJ`

- Delete line without saving in the buffer (black hole): `"_dd`

## Use registers
- Show the contents of registers: `:reg`
- Delete and save in 0 registers (from 0 to 9): `0_d`
- Copy and save in 0 registers: `0_y`
- Paste content of register 0: `0_p`

## Files
- Write the range to a file: `:[range]w FILE`
- Read file and paste it in the current spot: `:r FILE`

## Format / Indent
- Indent a range of lines: `:[range]!indent (or =)`
- Indent entire file: `gg=G`

- Visual select entire file: `ggVG`
- Format comments and text: `gq{motion}`
  - E.g., `gq)`

- Sort entire file: `%! sort | uniq`

# Buffers

- Redraw the screen: `:redraw`

- Next/previous file in the list: `:n, :N`
- Show all the files splitting the screen: `:all`
- Close all the previews: `CTRL-w + z`
- Show all the files in a tabbed way: `:tab ball`
- Close the tabs (with all the splits): `:tabc`

- Show all the buffers / files: `:buffers`

- Go to buffer N: `:b <NUMBER>`

- Go to previous / next buffer: `:bN`, `:bn`

- Go to buffer by name: `:buf ...` (`<CTRL-D>` to autocomplete)

- Print the name of the buffer: `:echo bufname("%")`

- Delete current buffer: `:bd`

- Delete buffer N: `:bd N`

## BufExplorer
- Get help about bufexplorer: `:help bufexplorer`
- To show the buffer browser: `\be or CTRL-j`

## MiniBufExplorer
- Select next/previous buffer: `CTRL-j and CTRL-k`
- To toggle MiniBufExplorer: `:TMiniBufExplorer`

# Viewports

- Split horizontally: `:split`, `:sp`

- Split vertically: `:vsplit`, `:vsp`

- Close the active window: `CTRL-w q`

## Move between viewports
- `CTRL-w CTRL-w`
- Move one viewport down: `CTRL-w j`
- Move one viewport up: `CTRL-w k`
- Move one viewport to the left: `CTRL-w h`
- Move one viewport to the right: `CTRL-w l`

- Resize viewports to be of equal size: `CTRL-w =`

## Change size of viewport
- Reduce active viewport by one line: `CTRL-w -`
- Reduce active viewport by 10 lines: `CTRL-w 10-`
- Increase active viewport by one line: `CTRL-w +`

# Search and replace

- Clear previous search: `:noh`, `:nohlsearch`

## Turn off highlighting completely
- `:set nohlsearch`
- Toggle `:set nohlsearch!`

- Help with search and replace: `:help substitute`

## Replace OLD with NEW everywhere
- `:%s/OLD/NEW/g`
- Case insensitive: `cgi`
- Case sensitive: `cg` or `cgI`

- Incremental search: `/`

- To replace exactly the word foo: `%s/\<foo\>/bar/gc`

## To replace from the cursor to the end of the file
- `.,$s/foo/bar/`
- The range is `.` (cursor) to `$`

## Repeat last substitute
- `&` (flags are not remembered)
- `&&` (to keep the flags)

- Repeat last substitute on all lines with the same flags: `g&`

## Pull <word> onto search/command line
- `/<CTRL-r><CTRL-w>`
- `:%s/<CTRL-r><CTRL-w>/NEW/g`

- Delete using non-greedy matching: `:%s/|\/tmp.\{-}|/|/g`

- Preview search result at the bottom of the window: `:g/\<DmaImage2System\>/`

- To show some context: `:g/\<DmaImage2System\>/z#.5 " echo "=========="`

- Preview all the occurrence of the word under cursor in a separate window: `[I`

# Navigating Filesystem

- `netrw` ships with Vim and allows to navigate the file system

- Open netrw in the directory of the current file: `:Ex`
- Open current path: `:Ex <path>`
- Open horizontal split: `:Sex`
- Open vertical split: `:Vex`

- Open file or descent into dir: `<Enter>`
- Go up one dir: `-`
- Go back to previous dir (history): `u`
- Preview file in a split: `p`

# Vim regex

- Help for regex: `:help regex`

## Using perl
- Execute perl CMD on all the lines selected: `:perldo CMD`

- Print matching lines: `:%!perl -ne 'print if ///'`
- Replace and print: `:%!perl -ne 's/reg[^\w]//; print $_'`

- Use very magic (to use perl-like tokens): `/\vdassert_.*\(/`

- Alphabetic character: `\a`
- Save token: `\( \)`
- Token number 1: `\1`
- Any character: `.`
- One or more: `\+`
- Any number of times: `*`
- a or b: `[ab]`
- Match foo or bar: `foo\|bar`
- Match the previous token non greedy (e.g., .{-}): `{-}`
- Like `*` but not greedy so we can match multiple times the same line: `\{-0,}`
- Replace `1.` with `5$1`: `s/1\(.\)/5\1/cg`

## Vim regex examples
- dos2unix (`<CTRL-v><CR>` to enter `^M` character): `:1,$s/^M//g`
- To add a `#` in front to a selection: `:'<,'>s/^\(.\)/#\1/g`
- To prepend a XXXX in front of each line of a block: `:'<,'>s/^/XXXX/`
- Replace `//... Ppe` with `//... PPE`: `:%s/\(\/\/.*\)Ppe/\1PPE/cg`
- Replace `/* XXXX */` with `// XXXX`: `:%s/\/\*\(.*\)\*\//\/\/\1/cg`

# Tags

- Show tag stack: `:tags`
- Jump to tag: `:tag TAG`
- List matching tags: `:ts[elect] TAG`
- Split window `:tag`: `:stag TAG`

## Jump to the tag under the cursor
- `CTRL-]`
- Use `:tselect` instead of :tag: `g]`

## Preview / split tag
- Split horizontally and get to the tag under the cursor: `CTRL-W ]`
- Preview window for the tag under cursor: `CTRL-W g}`
- Close all the preview windows: `CTRL-W z`

- Pop from the stack: `CTRL-t`

# Indent

- Get help: `:help cinoptions, :help cinoptions-values`
- To reindent the current block: `=a{`

# Quickfix

- Help quickfix: `:help quickfix`

## Make
```
:let &makeprg="cd /home/saggese/src/zs/trunks/tree2/build/host/; make"
```

## Grep and quickfix
```bash
> \grep -n -R "dbg.dassert" | tee /tmp/tmp
> vi -c "cfile /tmp/tmp"
```

## Quickfix: open window with all matches
- `:copen`
- Jump to n-th error: `:cc n`

## Go to prev / next quickfix
- Prev: `:cp` or `<F9>`
- Next: `:cn` or `<F10>`
- Show current error: `:cc`

# Marks

- List all the current marks: `:marks`

## Set a mark
- `m[a-z]`
- Marks valid within on file: `a-z`
- Marks (aka file marks) are valid between files: `A-Z`

- Go to the corresponding parenthesis: `%`

- Jump to beginning of line of mark: `'a`

- Jump to position of mark: `` `a ``

## Delete mark
- `:delmarks A`
- Delete all the marks: `:delmarks A-Z 0-9 a-z [].^\"` or `KillMarks`

- Beginning-end sentence markers: `()`

- Beginning-end paragraph markers: `{}`

- Alternate between last marks: `` ` ``

# Autocompletion

// http://www.vim.org/tips/tip.php?tip_id=91

## To find the matching words in insertion mode
- find matching word backwards: `<CTRL-P>`
- find matching word forward: `<CTRL-N>`

<!--
- :help i_Ctrl-N
- :help i_Ctrl-P
- :help ins-completion
- :help complete
- :help completeopt         " To change pop-up menu
-->

# Vimdiff

- vimdiff: go to prev difference: `[c`, `<F9>`
- vimdiff: go to next difference: `]c`, `<F10>`
- vimdiff: get text: `:diffget`
- vimdiff: copy next text: `:diffput`
- vimdiff: refresh update: `:diffupdate`

## vimdiff: Switch between vertical and horizontal split
- Vertical split: `CTRL-w J`
- Horizontal split: `CTRL-w H`

# SpellChecking

- To set the file: `:setlocal spelllang=en_us`, `:set spellfile=~/.vim/spell/en.utf-8.add`
- To enable / disable the spelling: `:setlocal spell / nospell`
- To go to the next/prev misspelled word: `]s`, `[s`
- To add to the local spell check: `zg`
- To see spelling suggestions: `z=`
- Accept the first suggestion: `1z=`
- To regenerate the .spl file after editing the .add file: `:mkspell! ~/.vim/spell/en.utf-8.add`

# Folding

- Open all folds: `zR`
- Close all folds: `zM`
- Open current fold recursively: `zO`

- Move the cursor to the next fold: `zj`
- Move the cursor to the previous fold: `zk`
- Jump to start/end of the fold: `[z, ]z`

- Toggle fold: `za`
- Toggle fold recursively: `zA`

# Crypt

- Open file
- Set `:cm blowfish2` and enter password
- Save `:w`

# Functions

- How to write functions: http://vim.sourceforge.net/tips/tip.php?tip_id=32
- Get help for functions: `:help functions`
- Get help for script: `:help script`
- http://vim.sourceforge.net/scripts/script.php?script_id=72
- http://vim.sourceforge.net/scripts/script.php?script_id=197

# Non-Ascii Editing

## Inserting non-printable chars

- To insert a tab, i.e., `^I`: `CTRL-v + TAB`
- To insert a return, i.e., `^M`: `CTRL-v + ENTER`
- To match non-ascii chars: `/[^\x00-\x7F]`

## Hex editing

- To see character under the cursor: `ga`
- To show non-printable characters: `:set display=uhex`
- To open a file in hex form: vim -b datafile, :%!xxd

## Digraphs

//Greek/math symbols in gvim with
//  http://www.alecjacobson.com/weblog/?p=443
//  http://tlt.its.psu.edu/suggestions/international/bylanguage/mathchart.html

- to see all the symbols: `:digraphs`

- triple equality: `CTRL-k =3`
- belongs: `CTRL-k (-`
- there exists: `CTRL-k TE`
- for all: `CTRL-k FA`

# Useful Packages

<!--
- Minibufexplorer
- Commentify
- Showmarks
- NERDTree

- Orgmode for vim
    - https://github.com/jceb/vim-orgmode

- Txtfmt (The Vim Highlighter) : "Rich text" highlighting in Vim! (colors,
  underline, bold, italic, etc...)
    - `http://www.vim.org/scripts/script.php?script_id=2208`

- MultipleSearch : Highlight multiple searches at the same time, each with a
  different color.
     - `http://www.vim.org/scripts/script.php?script_id=479`

- project.tar.gz : Organize/Navigate projects of files (like IDE/buffer explorer)
    - `http://www.vim.org/scripts/script.php?script_id=69`

- DoxygenToolkit.vim : Simplify Doxygen documentation in C, C++, Python.
    - `http://www.vim.org/scripts/script.php?script_id=987`
-->

## Commentify

- Comment the selected block out with: `:norm i# (lower case i)`
- To uncomment, highlight your block again, and uncomment with: `:norm ^x`
- Toggle: `:TC`
- Comment: `:CC`
- Uncomment: `:UC`
- To prepend a XXXX in front of each line of a block: `:'<,'>s/^/XXXX/`

## vim-commentary

- Plugin: `tpope/vim-commentary` (https://github.com/tpope/vim-commentary)
- Install via vim-plug: add `Plug 'tpope/vim-commentary'` to `.vimrc`, then run
  `:PlugInstall`
- Comment/uncomment current line: `gcc`
- Comment/uncomment a motion or visual selection: `gc{motion}`, e.g. `gcap`,
  or `gc` in visual mode

<!--
## CTRLP

// https://vimawesome.com/plugin/ctrlp-vim-everything-has-changed

- Fuzzy, buffer, mru, tag finder
- To find a font

## VimTaglist

// http://vim-taglist.sourceforge.net/manual.html
:TlistToggle
:TlistShowPrototype (CTRL-SPACE or space on the tag window)
:help TlistToggle
-->

## ctags

- Jump to the tag underneath the cursor: `g]` or `:tag CTRL+(gr)`
- Jump back up in the tag stack: `Ctrl-t`
- Jump to next / prev matching tag: `:tn`, `:tp`
- List the tags that match <tag_name>: `:ts[elect] <tag_name>`

- Preview window (horizontal split): `:pts[elect] <tag_name>`
- Open a preview window with the location of the tag definition: `Ctrl + w }`
- Close preview window: `:pc`
- Show the contents of the tag stack: `:tags`

- Copy word under cursor to command line: `CTRL + (gr)`

## Alternate
- Switches to the file corresponding to the current file being edited: `:A`
- Splits and switches: `:AS`
- Vertical splits and switches: `:AV`
- New tab and switches: `:AT`
- Cycles through matches: `:AN`
- Switches to file under cursor: `:IH`
- Splits and switches: `:IHS`
- Vertical splits and switches: `:IHV`
- New tab and switches: `:IHT`
- Cycles through matches: `:IHN`
- Switches to file under cursor: `<Leader>ih`
- Switches to the alternate file of file under cursor: `<Leader>is`
- (e.g. on <foo.h> switches to foo.cpp): ``
- Cycles through matches: `<Leader>ihn`

## ShowMarks

- Help about marks: `:tab help showmarks`
- Toggles ShowMarks on and off.: `\mt`
- Hides an individual mark.: `\mh`
- Hides all marks in the current buffer.: `\ma`
- Places the next available mark.: `\mm`

## Cscope

<!--
:cs help
If you get "cscope no connections", use :cs add cscope.out
:cs reset                 " To reset the connections

CTRL-\ + char:
  s " Symbol: find all references to the token under cursor
  g " Global: find global definition(s) of the token under cursor
  c " Calls: find all calls to the function name under cursor
  t " Text: find all instances of the text under cursor
  e " Egrep: egrep search for the word under cursor
  f " File: open the filename under cursor
  i " Includes: find files that include the filename under cursor
  d " Called: find functions that function under cursor calls
-->
