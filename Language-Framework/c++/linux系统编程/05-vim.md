# VIM

## 三种工作模式

* 命令模式：`编辑模式下 按 esc 进入， 命令行模式 双击 esc 回到命令模式`
* 命令行模式 : `命令模式下 输入 : 进入`
* 编辑模式：`命令模式下 输入 i 进入`



**命令行模式**

* ​



**命令模式**

* 光标的移动
  * `h 前,j 下,k 上,l 后`
  * `0 移动到头部，$移动到当前行 尾部`
  * `gg 移动到文件的开始位置, GG移动到文件的尾部`
  * `500G 移动到第500行`
* 删除，其实是剪切
  * `X 删除光标前面的字符， x 删除光标后面的字符`
  * `dw 删除下一个位置到第一个空格位置，删除word`
  * `d0 光标到行首， D删除到行尾`
  * `dd 删除当前行， 4dd 删除四行（当前光标所在行开始删）`
  * `u` 撤销操作，反撤销 `ctrl+r`
* 复制/粘贴
  * `p : 粘贴， P光标所在行上边`
  * `yy : 复制当前行，4yy：复制四行`
* 可视模式： **v** , 选中部分内容的复制粘贴。
  * 先移动光标，然后进入可视模式，然后选择想要操作的部分
  * `d：删除，y: 复制，p：粘贴`
* 文本查找：
  * 先按下 `/` 然后敲入想要找的 文本 `n:向下走，N:向上走` （当前光标向下查找）
  * `?` 当前光标位置向上查找
  * 将光标移动到 单词上，然后按 `#` 查找当前光标所在单词
* 缩进处理
  * `>>` 右缩进
  * `<<` 左缩进
* 使用 man 文档
  * 选中函数名，然后 按 `number K` number指定API在第几章



**命令模式--->文本模式**

* `a`: 插入到光标的后面
* `A`： 直接跳到行尾，然后插入
* `i`： 插入到光标前面
* `I` ：跳到行首
* `o`：光标所在位置向下开一行
* `O`：光标所在位置向上开一行
* `s` ：删除光标后面字符，然后可以插入
* `S`：删除光标所在行，然后插入



**命令行模型的操作**

* 行跳转：直接输入 `300` 回车就能跳到 300 行
* 字符传替换：
  * 按行操作：移动光标到指定行，然后进入命令行模式
    * `:s/jack/tom` 替换第一个 `jack` 到 `tom`
    * `:s/jack/tom/g` 此行所有 `jack` 到 `tom`
  * 所有文件操作
    * `:%s/jack/tom` :替换所有行第一个 jack 到 tom
    * `:%s/jack/tom/g` : 替换所有
    * `:10,10s/jack/tom/g` :第10行到 20行的所有 jack 到 tom
* 执行 shell 命令
  * `:!ls` ，感叹号+命令，然后回车就会执行了





**分屏操作，命令行模式下执行某命令**

* 水平分屏 `:sp` ，切换：`ctrl+ww`
* 垂直分屏 `vsp` ，`vsp filename` 。
* 关掉： `:q，:qall，wqall` 



**配置文件**

* 系统级配置文件： `/etc/vim/vimrc`
* 用户级别配置文件：`~/.vim/vimrc`

**关于<leader>**
[https://mounui.com/298.html](https://mounui.com/298.html)

**常用的vimrc**
```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Vundle:
"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set runtimepath+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'gmarik/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
Plugin 'flazz/vim-colorschemes'
Plugin 'vim-scripts/YankRing.vim'
Plugin 'vim-scripts/bufexplorer.zip'
Plugin 'scrooloose/nerdtree'
Plugin 'kien/ctrlp.vim'
Plugin 'vim-scripts/taglist.vim'
Plugin 'terryma/vim-expand-region'
Plugin 'bling/vim-airline'
Plugin 'Valloric/YouCompleteMe'
" Plugin 'godlygeek/tabular'
Plugin 'mileszs/ack.vim'
Plugin 'wincent/command-t'

" plugin from http://vim-scripts.org/vim/scripts.html
"Plugin 'L9'
" Git plugin not hosted on GitHub
"Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
"Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
"Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Avoid a name conflict with L9
"Plugin 'user/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this lin

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Basic Settings:
"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Sets how many lines of history VIM has to remember
set history=3000
set number

" Enable filetype plugins
filetype plugin on
filetype indent on

" Set to auto read when a file is changed from the outside
set autoread

" With a map leader it's possible to do extra key combinations
" like <leader>w saves the current file
let mapleader = ","
let g:mapleader = ","

" :W sudo saves the file
" (useful for handling the permission-denied error)
" command W w !sudo tee % > /dev/null

" Turn on the WiLd menu
set wildmenu
" Ignore compiled files
set wildignore=*.o,*~,*.pyc
if has("win16") || has("win32")
    set wildignore+=*/.git/*,*/.hg/*,*/.svn/*,*/.DS_Store
else
    set wildignore+=.git\*,.hg\*,.svn\*
endif

" Always shwo current position
set ruler

" Height of the command bar
set cmdheight=2

" A buffer becomes hidden when it is abandoned
set hid

" Configure backspace so it acts as it should act
set backspace=eol,start,indent
set whichwrap+=<,>,h,l

" In many terminal emulators the mouse works just fine, thus enable it.
"if has('mouse')
"  set mouse=a
"endif

" Ignore case when searching
set ignorecase

" When searching try to be smart about cases
set smartcase

" Highlight search results
set hlsearch

" Makes search act like search in modern browsers
set incsearch

" No annoying sound on errors
set noerrorbells
set novisualbell
set t_vb=
set tm=500

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Colors and Fonts
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Enable syntax highlighting
syntax enable

" Set extra options when running in GUI mode
if has("gui_running")
"    set guioptions-=T
"    set guioptions-=e
"    set t_Co=256
"    set guitablabel=%M\ %t
endif

" Set colorscheme to solarized dark
set background=dark
colorscheme molokai
" colorscheme solarized

" Set guifont to Monaco 14 pound
set guifont=Monaco:h14

" Set utf8 as standard encoding and en_US as the standard language
set encoding=utf8

" Use Unix as the standard file type
set ffs=unix,dos,mac

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Files, backups and undo
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Turn backup off, since most stuff is in SVN, git et.c anyway...
set nobackup
set nowb
set noswapfile

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Text, tab and indent related
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Use spaces instead of tabs
" set expandtab

" Be smart when using tabs ;)
set smarttab

" 1 tab == 4 spaces
set shiftwidth=4
set tabstop=4
set softtabstop=4

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Moving around, tabs, windows and buffers
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Treat long lines as break lines (useful when moving around in them)
map j gj
map k gk

" Disable highlight when <leader><cr> is pressed
map <silent> <leader><cr> :noh<cr>

" Smart way to move between windows
map <C-j> <C-W>j
map <C-k> <C-W>k
map <C-h> <C-W>h
map <C-l> <C-W>l

" map C-o to C-i
" map <C-i> <C-o>


" Useful mappings for managing tabs
map <leader>tn :tabnew<cr>
map <leader>to :tabonly<cr>
map <leader>tc :tabclose<cr>
map <leader>tm :tabmove
map <leader>t<leader> :tabnext<cr>

" Opens a new tab with the current buffer's path
" Super useful when editing files in the same directory
map <leader>te :tabedit <c-r>=expand("%:p:h")<cr>/

" Switch CWD to the directory of the open buffer
map <leader>cd :cd %:p:h<cr>:pwd<cr>

" Return to last edit position when opening files (You want this!)
autocmd BufReadPost *
     \ if line("'\"") > 0 && line("'\"") <= line("$") |
     \   exe "normal! g`\"" |
     \ endif
" Remember info about open buffers on close
set viminfo^=%

""""""""""""""""""""""""""""""
" => Status line
""""""""""""""""""""""""""""""
" Always show the status line
set laststatus=2

" Format the status line
" set statusline=\ %{HasPaste()}%F%m%r%h\ %w\ \ CWD:\ %r%{getcwd()}%h\ \ \ Line:\ %l

" Returns true if paste mode is enabled
function! HasPaste()
    if &paste
        return 'PASTE MODE  '
    en
    return ''
endfunction

" Move a line of text using ALT+[jk] or Comamnd+[jk] on mac
nmap <M-j> mz:m+<cr>`z
nmap <M-k> mz:m-2<cr>`z
vmap <M-j> :m'>+<cr>`<my`>mzgv`yo`z
vmap <M-k> :m'<-2<cr>`>my`<mzgv`yo`z

if has("mac") || has("macunix")
  nmap <D-j> <M-j>
  nmap <D-k> <M-k>
  vmap <D-j> <M-j>
  vmap <D-k> <M-k>
endif

" Open MacVim in fullscreen mode
if has("gui_macvim")
    set fuoptions=maxvert,maxhorz
    " au GUIEnter * set fullscreen
endif

" Disable scrollbars (real hackers don't use scrollbars for navigation!)
set guioptions-=r
set guioptions-=R
set guioptions-=l
set guioptions-=L

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Fast editing and reloading of vimrc configs
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
map <leader>e :e! ~/.vimrc<cr>
autocmd! bufwritepost vimrc source ~/.vimrc

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Turn persistent undo on
"    means that you can undo even when you close a buffer/VIM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
try
    set undodir=~/.vim/temp_dirs/undodir
    set undofile
catch
endtry

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Command mode related
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Bash like keys for the command line
cnoremap <C-A>		<Home>
cnoremap <C-E>		<End>
cnoremap <C-K>		<C-U>

cnoremap <C-P> <Up>
cnoremap <C-N> <Down>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Plugin Settings:
"
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""
" => bufExplorer plugin
""""""""""""""""""""""""""""""
let g:bufExplorerDefaultHelp=0
let g:bufExplorerShowRelativePath=1
let g:bufExplorerFindActive=1
let g:bufExplorerSortBy='name'
map <leader>o :BufExplorer<cr>

""""""""""""""""""""""""""""""
" => YankRing
""""""""""""""""""""""""""""""
let g:yankring_history_dir = '~/.vim/temp_dirs'
let g:yankring_max_element_length=104857600
nnoremap <silent> <leader>r :YRShow<CR>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Tlist
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
noremap <leader>t :TlistToggle<cr>

""""""""""""""""""""""""""""""
" => YouCompleteMe
""""""""""""""""""""""""""""""
let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'
let g:ycm_confirm_extra_conf= 0
let g:ycm_autoclose_preview_window_after_completion = 1
let g:ycm_key_invoke_completion = '<leader>c'
nnoremap <leader>jd :YcmCompleter GoToDefinitionElseDeclaration<cr>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Nerd Tree
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
let g:NERDTreeWinPos = "right"
let NERDTreeIgnore = ['\.pyc$', '\.o$', 'tags', 'cscope.*$']
let g:NERDTreeWinSize=35
map <leader>nn :NERDTreeToggle<cr>
map <leader>nb :NERDTreeFromBookmark
map <leader>nf :NERDTreeFind<cr>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Encoding related
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set encoding=utf-8
set fileencodings=ucs-bom,utf-8,cp936,gb18030,big5,euc-jp,euc-kr,latin1
set termencoding=utf-8

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Whitespace related settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Delete trailing white spaces
"function! <SID>StripTrailingWhitespaces()
    " Preparation: save last search, and cursor position.
""    let _s=@/
""    let l = line(".")
""    let c = col(".")
    " Do the business:
""    %s/\s\+$//e
    " Clean up: restore previous search history, and cursor position
""    let @/=_s
""    call cursor(l, c)
""endfunction
" map F5 to do the trick
""nnoremap <silent> <F5> :call <SID>StripTrailingWhitespaces()<CR>
" Automatic delete trailing space for py and js files
""autocmd BufWritePre *.py,*.js,*.cpp,*.c,*h :call <SID>StripTrailingWhitespaces()

" Shortcut to rapidly toggle `set list`
nmap <leader>l :set list!<CR>
"
" Use the same symbols as TextMate for tabstops and EOLs
" set listchars=tab:▸\ ,eol:¬

" Shortcut to toggle paste
set pastetoggle=<F2>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => cursor related settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set cursorcolumn cursorline
set colorcolumn=81

" map jk to <ESC>
inoremap jk <ESC>

"""""""""""""""""""""""""
" enable code folding
"""""""""""""""""""""""""
set foldmethod=indent
set foldlevel=99
execute pathogen#infect()
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0
nnoremap <space> za
inoremap ' ''<ESC>i
inoremap " ""<ESC>i
inoremap ( ()<ESC>i
inoremap [ []<ESC>i
inoremap { {<CR>}<ESC>O
```

