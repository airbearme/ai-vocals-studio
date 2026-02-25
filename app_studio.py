#!/usr/bin/env python3
"""
AI Vocals Studio Pro v3.0
Dark neon UI · Scrollable pages · Model preview · Organized output
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, json, threading, time, math, random, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ─────────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────────
C = {
    'bg':     '#080812',
    'bg2':    '#0d0d1e',
    'bg3':    '#12122a',
    'card':   '#16163a',
    'border': '#222255',
    'cyan':   '#00d4ff',
    'purple': '#9b5de5',
    'green':  '#05ffa1',
    'orange': '#ff9500',
    'red':    '#ff2d55',
    'pink':   '#ff006e',
    'white':  '#eef0ff',
    'gray':   '#7777aa',
    'dim':    '#333366',
}

FONT = {
    'h1':   ('Courier New', 20, 'bold'),
    'h2':   ('Courier New', 13, 'bold'),
    'h3':   ('Courier New', 11, 'bold'),
    'body': ('Courier New', 10),
    'sm':   ('Courier New', 9),
    'mono': ('Courier New', 10),
}

VOICE_PERSONAS = {
    '2pac':               {'pitch': -3,   'speed': 1.08, 'reverb': 0.35, 'gain': 3,  'desc': 'West Coast rap – deep, raw authority'},
    '2pac_custom_voice':  {'pitch': -3,   'speed': 1.08, 'reverb': 0.35, 'gain': 3,  'desc': 'Custom trained 2Pac voice'},
    '2pac_enhanced_voice':{'pitch': -3.5, 'speed': 1.10, 'reverb': 0.40, 'gain': 4,  'desc': 'Enhanced 2Pac – maximum realism'},
    'male':               {'pitch': -4,   'speed': 1.00, 'reverb': 0.20, 'gain': 2,  'desc': 'Generic deep male voice'},
    'female':             {'pitch': 4,    'speed': 1.00, 'reverb': 0.15, 'gain': 0,  'desc': 'Generic bright female voice'},
    'robot':              {'pitch': 0,    'speed': 0.88, 'reverb': 0.50, 'gain': 6,  'desc': 'Vocoder / robot effect'},
    'default':            {'pitch': 0,    'speed': 1.00, 'reverb': 0.05, 'gain': 0,  'desc': 'Clean unmodified voice'},
}

# ─────────────────────────────────────────────────────────────────
#  SCROLLABLE FRAME
# ─────────────────────────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, **kw):
        kw.setdefault('bg', C['bg'])
        super().__init__(parent, **kw)
        bg = kw['bg']

        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._sb = tk.Scrollbar(self, orient='vertical', command=self._canvas.yview,
                                bg=C['bg3'], troughcolor=C['bg2'], bd=0, width=10)
        self.inner = tk.Frame(self._canvas, bg=bg)

        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self._canvas.configure(yscrollcommand=self._sb.set)

        self._sb.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        self.inner.bind('<Configure>', self._on_inner_resize)
        self._canvas.bind('<Configure>', self._on_canvas_resize)

        # Scroll binding (only when hovered)
        self._canvas.bind('<Enter>', self._bind_scroll)
        self._canvas.bind('<Leave>', self._unbind_scroll)
        self.inner.bind('<Enter>', self._bind_scroll)

    def _on_inner_resize(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_resize(self, e):
        self._canvas.itemconfig(self._win, width=e.width)

    def _bind_scroll(self, _e=None):
        self._canvas.bind_all('<MouseWheel>', self._scroll)
        self._canvas.bind_all('<Button-4>', self._scroll)
        self._canvas.bind_all('<Button-5>', self._scroll)

    def _unbind_scroll(self, _e=None):
        self._canvas.unbind_all('<MouseWheel>')
        self._canvas.unbind_all('<Button-4>')
        self._canvas.unbind_all('<Button-5>')

    def _scroll(self, e):
        if e.num == 4:   self._canvas.yview_scroll(-2, 'units')
        elif e.num == 5: self._canvas.yview_scroll(2, 'units')
        else:            self._canvas.yview_scroll(int(-1*(e.delta/120)), 'units')

# ─────────────────────────────────────────────────────────────────
#  NEON BUTTON  (Canvas-based with glow)
# ─────────────────────────────────────────────────────────────────
class NeonBtn(tk.Canvas):
    def __init__(self, parent, text, cmd=None, color=None, w=160, h=38, font=None, **kw):
        color = color or C['cyan']
        kw.setdefault('bg', parent.cget('bg') if hasattr(parent, 'cget') else C['card'])
        super().__init__(parent, width=w, height=h, highlightthickness=0, bd=0, **kw)
        self.text, self.cmd, self.color = text, cmd, color
        self.W, self.H = w, h
        self.font = font or FONT['body']
        self._hover = False
        self._phase = 0
        self._draw()
        self.bind('<Enter>', lambda _: self._set_hover(True))
        self.bind('<Leave>', lambda _: self._set_hover(False))
        self.bind('<Button-1>', lambda _: cmd() if cmd else None)
        self._tick()

    def _set_hover(self, v):
        self._hover = v
        self._draw()

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _blend(self, hex_c, alpha):
        fr,fg,fb = int(hex_c[1:3],16),int(hex_c[3:5],16),int(hex_c[5:7],16)
        br,bg_v,bb = 8,8,18
        r=int(fr*alpha+br*(1-alpha)); g=int(fg*alpha+bg_v*(1-alpha)); b=int(fb*alpha+bb*(1-alpha))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _draw(self):
        self.delete('all')
        w, h = self.W, self.H
        pulse = 0.5 + 0.5*math.sin(self._phase) if self._hover else 0
        # outer glow layers
        if self._hover:
            for i in range(6, 0, -1):
                a = pulse * 0.35 / i
                gc = self._blend(self.color, a)
                self._rrect(i*1.5, i*1.5, w-i*1.5, h-i*1.5, 9, fill=gc, outline='')
        # bg fill
        bg = self._blend(self.color, 0.12 if self._hover else 0.06)
        self._rrect(2, 2, w-2, h-2, 8, fill=bg, outline=self.color, width=1+(self._hover))
        # shine line
        if self._hover:
            shine_a = 0.15 + 0.10*pulse
            sc = self._blend('#ffffff', shine_a)
            self.create_line(10, 4, w-10, 4, fill=sc, width=1)
        # text shadow + text
        tc = C['white'] if self._hover else self.color
        self.create_text(w//2+1, h//2+1, text=self.text, fill='#00000077',
                         font=self.font, anchor='center')
        self.create_text(w//2, h//2, text=self.text, fill=tc,
                         font=self.font, anchor='center')

    def _tick(self):
        try:
            if not self.winfo_exists(): return
        except tk.TclError: return
        if self._hover:
            self._phase += 0.12
            self._draw()
        self.after(50, self._tick)

# ─────────────────────────────────────────────────────────────────
#  NEON PROGRESS BAR
# ─────────────────────────────────────────────────────────────────
class NeonBar(tk.Canvas):
    def __init__(self, parent, w=400, h=10, color=None, **kw):
        color = color or C['cyan']
        kw.setdefault('bg', C['bg2'])
        super().__init__(parent, width=w, height=h, highlightthickness=0, bd=0, **kw)
        self.W, self.H, self.color = w, h, color
        self.value = 0
        self._shimmer = 0
        self._draw()
        self._tick()

    def set(self, v):
        self.value = max(0, min(100, v))
        self._draw()

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        if x2-x1 < r*2: r = max(1,(x2-x1)//2)
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        self.delete('all')
        w,h = self.W, self.H
        r = h//2
        self._rrect(0,0,w,h,r, fill=C['bg3'], outline=C['border'])
        fw = max(0, int(w * self.value / 100))
        if fw > r*2:
            self._rrect(0,0,fw,h,r, fill=self.color, outline='')
            # shimmer
            sx = self._shimmer % fw
            sw = min(40, fw)
            if sx + sw < fw:
                self.create_rectangle(sx, 1, sx+sw, h-1,
                                      fill='white', outline='', stipple='gray25')
            # tip glow
            self.create_oval(fw-h, 0, fw+h, h, fill=self.color, outline='')
            self.create_oval(fw-r+1, 1, fw+r-1, h-1, fill='white', outline='')

    def _tick(self):
        try:
            if not self.winfo_exists(): return
        except tk.TclError: return
        if 0 < self.value < 100:
            self._shimmer += 4
            self._draw()
        self.after(40, self._tick)

# ─────────────────────────────────────────────────────────────────
#  WAVEFORM DISPLAY
# ─────────────────────────────────────────────────────────────────
class Waveform(tk.Canvas):
    def __init__(self, parent, w=560, h=72, **kw):
        kw.setdefault('bg', C['bg3'])
        super().__init__(parent, width=w, height=h,
                         highlightthickness=1, highlightbackground=C['border'], **kw)
        self.W, self.H = w, h
        self._bars = []
        self._phase = 0
        self._active = False
        self._idle()
        self._tick()

    def _idle(self):
        self.delete('all')
        mid = self.H // 2
        self.create_line(10, mid, self.W-10, mid, fill=C['dim'], width=1)
        self.create_text(self.W//2, mid, text='NO AUDIO',
                         fill=C['dim'], font=FONT['sm'])

    def load(self, path):
        try:
            if HAS_LIBROSA:
                y, _ = librosa.load(str(path), sr=None, mono=True, duration=10)
                n = 80
                cs = max(1, len(y)//n)
                self._bars = [float(abs(y[i*cs:(i+1)*cs]).max()) for i in range(n)]
                mx = max(self._bars) or 1
                self._bars = [b/mx for b in self._bars]
            else:
                self._bars = [random.uniform(0.1, 0.9) for _ in range(80)]
            self._render()
        except Exception:
            self._idle()

    def set_active(self, v):
        self._active = v
        if not v and self._bars:
            self._render()
        elif not v:
            self._idle()

    def _render(self):
        self.delete('all')
        if not self._bars:
            self._idle()
            return
        w, h, n = self.W, self.H, len(self._bars)
        bw = w / n
        mid = h // 2
        for i, bar in enumerate(self._bars):
            x = i * bw
            pulse = (1.0 + 0.25*math.sin(self._phase + i*0.25)) if self._active else 1.0
            bh = max(2, bar * (h-6)/2 * pulse)
            t = i / n
            r = int(0 * (1-t) + 155 * t)
            g = int(212 * (1-t) + 93 * t)
            b = int(255)
            col = f'#{r:02x}{g:02x}{b:02x}'
            self.create_rectangle(x+1, mid-bh, x+bw-1, mid+bh,
                                  fill=col, outline='')
        if self._active:
            self.create_rectangle(0, 0, w, h, fill='', outline=C['cyan'], width=1)

    def _tick(self):
        try:
            if not self.winfo_exists(): return
        except tk.TclError: return
        if self._active:
            self._phase += 0.06
            self._render()
        self.after(50, self._tick)

# ─────────────────────────────────────────────────────────────────
#  VU METER
# ─────────────────────────────────────────────────────────────────
class VU(tk.Canvas):
    def __init__(self, parent, w=250, h=16, **kw):
        kw.setdefault('bg', C['bg2'])
        super().__init__(parent, width=w, height=h, highlightthickness=0, **kw)
        self.W, self.H = w, h
        self._lv = 0.0
        self._pk = 0.0
        self._ph = 0
        self._draw()
        self._tick()

    def push(self, v):
        self._lv = min(1.0, max(0, v))
        if self._lv > self._pk:
            self._pk = self._lv
            self._ph = 25

    def _draw(self):
        self.delete('all')
        n = 28
        bw = self.W / n
        for i in range(n):
            t = i / n
            col_on  = C['green'] if t < 0.6 else (C['orange'] if t < 0.85 else C['red'])
            col_off = '#003322' if t < 0.6 else ('#332200' if t < 0.85 else '#330011')
            is_peak = abs(self._pk - t) < (1/n) and self._ph > 0
            fill = '#ffffff' if is_peak else (col_on if self._lv > t else col_off)
            self.create_rectangle(i*bw+1, 2, (i+1)*bw-1, self.H-2, fill=fill, outline='')

    def _tick(self):
        try:
            if not self.winfo_exists(): return
        except tk.TclError: return
        self._lv *= 0.88
        if self._ph > 0:
            self._ph -= 1
            if self._ph == 0: self._pk = 0
        self._draw()
        self.after(45, self._tick)

# ─────────────────────────────────────────────────────────────────
#  SECTION HEADER / CARD helpers (standalone)
# ─────────────────────────────────────────────────────────────────
def section_hdr(parent, text, color=None, bg=None):
    color = color or C['cyan']
    bg    = bg    or C['bg']
    row = tk.Frame(parent, bg=bg)
    row.pack(fill='x', padx=12, pady=(14, 3))
    tk.Label(row, text=text, fg=color, bg=bg, font=FONT['h3']).pack(side='left')
    cv = tk.Canvas(row, height=1, bg=bg, highlightthickness=0)
    cv.pack(side='left', fill='x', expand=True, padx=(8, 0), pady=8)
    cv.create_line(0, 0, 3000, 0, fill=color)

def card(parent, bg=None):
    bg = bg or C['bg']
    outer = tk.Frame(parent, bg=C['card'],
                     highlightthickness=1, highlightbackground=C['border'])
    outer.pack(fill='x', padx=12, pady=4)
    inner = tk.Frame(outer, bg=C['card'])
    inner.pack(fill='x', padx=14, pady=10)
    return inner

def lbl(parent, text, color=None, font=None, **pk):
    color = color or C['gray']
    w = tk.Label(parent, text=text, fg=color, bg=parent.cget('bg'),
                 font=font or FONT['sm'])
    if pk: w.pack(**pk)
    return w

def styled_entry(parent, var, width=40):
    e = tk.Entry(parent, textvariable=var, bg=C['bg3'], fg=C['white'],
                 insertbackground=C['cyan'], font=FONT['body'], relief='flat',
                 highlightthickness=1, highlightcolor=C['cyan'],
                 highlightbackground=C['border'], width=width)
    e.pack(side='left', fill='x', expand=True, ipady=6)
    return e

def styled_text(parent, height=6, color=None):
    t = tk.Text(parent, height=height, wrap='word',
                bg=C['bg3'], fg=color or C['white'],
                insertbackground=C['cyan'], font=FONT['mono'],
                relief='flat', padx=10, pady=8,
                highlightthickness=1, highlightcolor=C['border'],
                highlightbackground=C['border'])
    sb = tk.Scrollbar(parent, orient='vertical', command=t.yview,
                      bg=C['bg3'], troughcolor=C['bg2'], bd=0, width=8)
    t.configure(yscrollcommand=sb.set)
    t.pack(side='left', fill='both', expand=True)
    sb.pack(side='right', fill='y')
    return t

def styled_combo(parent, var, values, width=20):
    cb = ttk.Combobox(parent, textvariable=var, values=values,
                      state='readonly', width=width, font=FONT['sm'])
    cb.pack(side='left', padx=(0, 10))
    return cb

# ─────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────
class StudioPro:
    def __init__(self, root):
        self.root = root
        self.root.title("AI VOCALS STUDIO PRO")
        self.root.geometry("1380x880")
        self.root.minsize(1000, 720)
        self.root.configure(bg=C['bg'])

        # ── dirs ────────────────────────────────────────────
        self.MODELS  = Path('models')
        self.DATASET = Path('dataset')
        self.OUTPUT  = Path('output')
        self.INPUT   = Path('input')
        for d in [self.MODELS, self.DATASET, self.OUTPUT, self.INPUT]:
            d.mkdir(exist_ok=True)

        # ── state ───────────────────────────────────────────
        self.audio_path_var   = tk.StringVar()
        self.model_var        = tk.StringVar()
        self.input_type_var   = tk.StringVar(value='text')
        self.tts_voice_var    = tk.StringVar(value='default')
        self.tts_speed_var    = tk.StringVar(value='1.0')
        self.train_model_var  = tk.StringVar()
        self.status_var       = tk.StringVar(value='READY')
        self.last_output      = None
        self._generating      = False
        self._preview_proc    = None

        self._hdr_phase = 0.0

        # ── build ───────────────────────────────────────────
        self._setup_ttk()
        self._build_header()
        self._build_nav()
        self._content = tk.Frame(self.root, bg=C['bg'])
        self._content.pack(fill='both', expand=True)
        self._pages = {}
        self._build_generate()
        self._build_models()
        self._build_training()
        self._build_outputs()
        self._build_footer()

        self._show('generate')
        self.root.after(300, self._refresh_models)
        self.root.after(50,  self._hdr_tick)

    # ─────────────────────────────────────────────────────────
    #  TTK STYLE
    # ─────────────────────────────────────────────────────────
    def _setup_ttk(self):
        s = ttk.Style(self.root)
        s.theme_use('clam')
        s.configure('TCombobox',
                    fieldbackground=C['bg3'], background=C['bg3'],
                    foreground=C['white'], selectbackground=C['purple'],
                    selectforeground=C['white'], bordercolor=C['border'],
                    arrowcolor=C['cyan'], insertcolor=C['white'])
        s.map('TCombobox', fieldbackground=[('readonly', C['bg3'])],
              foreground=[('readonly', C['white'])])
        s.configure('TScrollbar', background=C['bg3'], troughcolor=C['bg2'],
                    arrowcolor=C['gray'], bordercolor=C['border'])
        s.configure('TScale', background=C['card'], troughcolor=C['bg3'])

    # ─────────────────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────────────────
    def _build_header(self):
        self._hdr_cv = tk.Canvas(self.root, height=82, bg=C['bg'],
                                  highlightthickness=0)
        self._hdr_cv.pack(fill='x')

    def _hdr_tick(self):
        try:
            if not self.root.winfo_exists(): return
        except tk.TclError: return
        self._hdr_phase += 0.06
        cv = self._hdr_cv
        cv.delete('all')
        w = cv.winfo_width() or 1380
        h = 82
        # gradient bg
        for y in range(h):
            t = y / h
            r = int(8  + t*12)
            g = int(8  + t*4)
            b = int(18 + t*22)
            cv.create_line(0, y, w, y, fill=f'#{r:02x}{g:02x}{b:02x}')
        # animated glow wave at bottom
        ph = self._hdr_phase
        for x in range(0, w, 2):
            t = x / w
            wave = math.sin(t*12 + ph) * 3.5
            br = 0.45 + 0.45*math.sin(t*6 + ph*1.4)
            r2 = int(0 + 155*br)
            g2 = int(212*br)
            b2 = int(255*br)
            yy = h - 4 + wave
            cv.create_oval(x, yy, x+2, yy+2, fill=f'#{r2:02x}{g2:02x}{b2:02x}', outline='')
        # title glow
        for off, alpha in [(3,'#00001a'),(2,'#000033'),(1,'#000055')]:
            cv.create_text(w//2+off, 38+off, text='🎤  AI VOCALS STUDIO PRO',
                           fill=alpha, font=FONT['h1'], anchor='center')
        cv.create_text(w//2, 38, text='🎤  AI VOCALS STUDIO PRO',
                       fill=C['cyan'], font=FONT['h1'], anchor='center')
        cv.create_text(w//2, 62, text='PROFESSIONAL VOICE SYNTHESIS ENGINE',
                       fill=C['purple'], font=FONT['sm'], anchor='center')
        # corner accents
        for cx, col in [(8, C['cyan']), (w-8, C['purple'])]:
            for r, a in [(10,0.15),(6,0.3),(3,0.7),(1.5,1.0)]:
                cv.create_oval(cx-r, 6-r, cx+r, 6+r,
                               fill=self._blend(col, a), outline='')
        self.root.after(45, self._hdr_tick)

    def _blend(self, hex_c, a):
        fr,fg,fb = int(hex_c[1:3],16),int(hex_c[3:5],16),int(hex_c[5:7],16)
        r=int(fr*a+8*(1-a)); g=int(fg*a+8*(1-a)); b=int(fb*a+18*(1-a))
        return f'#{r:02x}{g:02x}{b:02x}'

    # ─────────────────────────────────────────────────────────
    #  NAV BAR
    # ─────────────────────────────────────────────────────────
    def _build_nav(self):
        nav = tk.Frame(self.root, bg=C['bg2'], height=46)
        nav.pack(fill='x')
        nav.pack_propagate(False)
        tk.Canvas(nav, height=1, bg=C['border'], highlightthickness=0).pack(fill='x')
        center = tk.Frame(nav, bg=C['bg2'])
        center.pack(expand=True, fill='both')
        self._nav_btns = {}
        tabs = [
            ('generate', '⚡  GENERATE', C['cyan']),
            ('models',   '🤖  MODELS',   C['purple']),
            ('training', '🎓  TRAINING',  C['green']),
            ('outputs',  '📁  OUTPUTS',   C['orange']),
        ]
        self._cur_page = 'generate'
        for key, label, color in tabs:
            b = tk.Label(center, text=label, font=FONT['h3'],
                         fg=C['dim'], bg=C['bg2'], padx=22, pady=10, cursor='hand2')
            b.pack(side='left')
            b.bind('<Button-1>', lambda e, k=key: self._show(k))
            b.bind('<Enter>',   lambda e, b=b, c=color: b.config(fg=c))
            b.bind('<Leave>',   lambda e, b=b, k=key: b.config(
                fg=C['white'] if self._cur_page == k else C['dim']))
            self._nav_btns[key] = (b, color)
        tk.Canvas(nav, height=1, bg=C['border'], highlightthickness=0).pack(fill='x', side='bottom')

    def _show(self, key):
        for k, f in self._pages.items():
            f.pack_forget()
        self._pages[key].pack(fill='both', expand=True)
        self._cur_page = key
        for k, (b, col) in self._nav_btns.items():
            b.config(fg=col if k == key else C['dim'])

    # ═════════════════════════════════════════════════════════
    #  PAGE: GENERATE
    # ═════════════════════════════════════════════════════════
    def _build_generate(self):
        sf_frame = ScrollableFrame(self._content)
        self._pages['generate'] = sf_frame
        f = sf_frame.inner

        # ── Input type toggle ──────────────────────────────
        section_hdr(f, 'INPUT SOURCE', C['cyan'])
        c = card(f)
        rb_row = tk.Frame(c, bg=C['card'])
        rb_row.pack(fill='x', pady=4)
        for val, lbl_t in [('text','📝  TEXT TO SPEECH'), ('audio','🎵  AUDIO TO AUDIO')]:
            tk.Radiobutton(rb_row, text=lbl_t, variable=self.input_type_var,
                           value=val, command=self._toggle_input,
                           fg=C['cyan'], bg=C['card'], selectcolor=C['bg3'],
                           activeforeground=C['white'], activebackground=C['card'],
                           font=FONT['body']).pack(side='left', padx=16, pady=4)

        # ── Input container ────────────────────────────────
        self._input_container = tk.Frame(f, bg=C['bg'])
        self._input_container.pack(fill='x')

        # Text section
        self._txt_sec = tk.Frame(self._input_container, bg=C['bg'])
        section_hdr(self._txt_sec, 'TEXT INPUT', C['cyan'])
        tc = card(self._txt_sec)
        txt_wrap = tk.Frame(tc, bg=C['card'])
        txt_wrap.pack(fill='x')
        self._text_box = styled_text(txt_wrap, height=5)
        self._text_box.insert('1.0', 'Enter lyrics or speech here...')
        self._text_box.bind('<FocusIn>', self._clear_placeholder)
        ts_row = tk.Frame(tc, bg=C['card'])
        ts_row.pack(fill='x', pady=6)
        lbl(ts_row, 'VOICE:', side='left', padx=(0,6))
        styled_combo(ts_row, self.tts_voice_var,
                     ['default','male','female','robot'], width=12)
        lbl(ts_row, 'SPEED:', side='left', padx=(6,6))
        styled_combo(ts_row, self.tts_speed_var,
                     ['0.6','0.8','1.0','1.1','1.2','1.5'], width=8)

        # Audio section
        self._aud_sec = tk.Frame(self._input_container, bg=C['bg'])
        section_hdr(self._aud_sec, 'AUDIO INPUT', C['cyan'])
        ac = card(self._aud_sec)
        ar = tk.Frame(ac, bg=C['card'])
        ar.pack(fill='x', pady=4)
        styled_entry(ar, self.audio_path_var, width=45)
        NeonBtn(ar, 'BROWSE', cmd=self._select_audio,
                color=C['cyan'], w=100, h=34).pack(side='right', padx=(8,0))
        self._audio_wave = Waveform(ac, w=540, h=66)
        self._audio_wave.pack(fill='x', pady=6)

        # ── Model selection ────────────────────────────────
        section_hdr(f, 'VOICE MODEL', C['purple'])
        mc = card(f)
        mr = tk.Frame(mc, bg=C['card'])
        mr.pack(fill='x', pady=4)
        lbl(mr, 'MODEL:', side='left', padx=(0,8))
        self._gen_model_cb = styled_combo(mr, self.model_var, [], width=26)
        self._gen_model_cb.bind('<<ComboboxSelected>>', self._on_model_select)
        NeonBtn(mr, '▶ PREVIEW', cmd=self._preview_model,
                color=C['green'], w=120, h=34).pack(side='left', padx=(0,6))
        NeonBtn(mr, '■ STOP', cmd=self._stop_preview,
                color=C['red'], w=80, h=34).pack(side='left')
        self._model_info = lbl(mc, 'Select a model', color=C['gray'])
        self._model_info.pack(anchor='w', pady=(4,0))
        self._prev_wave = Waveform(mc, w=540, h=56)
        self._prev_wave.pack(fill='x', pady=6)

        # ── FX sliders ─────────────────────────────────────
        section_hdr(f, 'VOICE EFFECTS', C['orange'])
        fxc = card(f)
        fxg = tk.Frame(fxc, bg=C['card'])
        fxg.pack(fill='x', pady=4)
        self._fx = {}
        params = [
            ('PITCH',  'pitch', -8,  8,   0),
            ('SPEED',  'speed',  0.5, 2.0, 1.0),
            ('REVERB', 'reverb', 0,   1,   0.05),
            ('BASS',   'bass',   0.5, 2.5, 1.0),
        ]
        for i,(lbl_t, key, lo, hi, dflt) in enumerate(params):
            col = tk.Frame(fxg, bg=C['card'])
            col.grid(row=0, column=i, padx=18, pady=4, sticky='n')
            lbl(col, lbl_t, color=C['gray']).pack()
            var = tk.DoubleVar(value=dflt)
            self._fx[key] = var
            tk.Scale(col, variable=var, from_=lo, to=hi,
                     resolution=0.05 if key != 'pitch' else 0.5,
                     orient='vertical', length=90,
                     bg=C['card'], fg=C['gray'], troughcolor=C['bg3'],
                     activebackground=C['cyan'], highlightthickness=0,
                     font=FONT['sm']).pack()
            lbl(col, f'{dflt}', color=C['dim']).pack()

        # ── Generate button ────────────────────────────────
        section_hdr(f, 'GENERATE', C['green'])
        gc = card(f)
        gr = tk.Frame(gc, bg=C['card'])
        gr.pack(pady=8)
        NeonBtn(gr, '⚡  GENERATE VOCALS', cmd=self._generate,
                color=C['green'], w=240, h=52, font=FONT['h2']).pack(side='left', padx=10)
        NeonBtn(gr, '🔄  CLEAR LOG', cmd=lambda: self._out_log.delete('1.0','end'),
                color=C['dim'], w=130, h=52).pack(side='left')

        # ── Output ─────────────────────────────────────────
        section_hdr(f, 'OUTPUT', C['purple'])
        oc = card(f)
        self._out_wave = Waveform(oc, w=540, h=66)
        self._out_wave.pack(fill='x', pady=(0,6))
        vu_row = tk.Frame(oc, bg=C['card'])
        vu_row.pack(fill='x', pady=(0,6))
        self._vu = VU(vu_row, w=260, h=16)
        self._vu.pack(side='left')
        self._out_file_lbl = lbl(vu_row, 'No output yet', color=C['gray'])
        self._out_file_lbl.pack(side='right')
        log_wrap = tk.Frame(oc, bg=C['card'])
        log_wrap.pack(fill='x')
        self._out_log = styled_text(log_wrap, height=11, color=C['green'])
        obr = tk.Frame(oc, bg=C['card'])
        obr.pack(fill='x', pady=8)
        NeonBtn(obr, '▶ PLAY OUTPUT', cmd=self._play_output,
                color=C['green'], w=150, h=36).pack(side='left', padx=(0,8))
        NeonBtn(obr, '📁 OPEN FOLDER', cmd=self._open_out_folder,
                color=C['cyan'], w=150, h=36).pack(side='left')

        self._toggle_input()

    # ═════════════════════════════════════════════════════════
    #  PAGE: MODELS
    # ═════════════════════════════════════════════════════════
    def _build_models(self):
        sf_frame = ScrollableFrame(self._content)
        self._pages['models'] = sf_frame
        f = sf_frame.inner

        section_hdr(f, 'MODEL LIBRARY', C['purple'])
        cc = card(f)
        cr = tk.Frame(cc, bg=C['card'])
        cr.pack(fill='x', pady=6)
        NeonBtn(cr, '🔄 REFRESH', cmd=self._refresh_models,
                color=C['purple'], w=130, h=38).pack(side='left', padx=(0,8))
        NeonBtn(cr, '📥 IMPORT .PTH', cmd=self._import_model_file,
                color=C['orange'], w=150, h=38).pack(side='left', padx=(0,8))
        NeonBtn(cr, '📂 IMPORT FOLDER', cmd=self._import_model_folder,
                color=C['orange'], w=160, h=38).pack(side='left', padx=(0,8))
        NeonBtn(cr, '📁 OPEN DIR', cmd=lambda: self._open_dir(self.MODELS),
                color=C['dim'], w=120, h=38).pack(side='left')

        self._model_cards_host = tk.Frame(f, bg=C['bg'])
        self._model_cards_host.pack(fill='x')

    # ═════════════════════════════════════════════════════════
    #  PAGE: TRAINING
    # ═════════════════════════════════════════════════════════
    def _build_training(self):
        sf_frame = ScrollableFrame(self._content)
        self._pages['training'] = sf_frame
        f = sf_frame.inner

        section_hdr(f, 'TRAINING DATA MANAGER', C['green'])
        sc = card(f)
        sr = tk.Frame(sc, bg=C['card'])
        sr.pack(fill='x', pady=6)
        lbl(sr, 'TARGET MODEL:', side='left', padx=(0,8))
        self._train_cb = styled_combo(sr, self.train_model_var, [], width=22)
        self._train_cb.bind('<<ComboboxSelected>>', lambda _: self._refresh_ds())
        NeonBtn(sr, '+ NEW MODEL', cmd=self._new_model_dialog,
                color=C['green'], w=130, h=34).pack(side='left')

        section_hdr(f, 'IMPORT AUDIO', C['cyan'])
        ic = card(f)
        ir = tk.Frame(ic, bg=C['card'])
        ir.pack(fill='x', pady=6)
        NeonBtn(ir, '🎵 CLIP', cmd=self._import_clip,
                color=C['cyan'], w=100, h=36).pack(side='left', padx=(0,8))
        NeonBtn(ir, '📂 FOLDER', cmd=self._import_ds_folder,
                color=C['cyan'], w=110, h=36).pack(side='left', padx=(0,8))
        NeonBtn(ir, '📁 OPEN', cmd=self._open_ds_folder,
                color=C['dim'], w=100, h=36).pack(side='left')

        section_hdr(f, 'DATASET FILES', C['green'])
        dc = card(f)
        dw = tk.Frame(dc, bg=C['card'])
        dw.pack(fill='x')
        self._ds_text = styled_text(dw, height=18, color=C['green'])

        section_hdr(f, 'VOICE ANALYSIS', C['orange'])
        vc = card(f)
        NeonBtn(vc, '🔬 ANALYZE VOICE', cmd=self._analyze_voice,
                color=C['orange'], w=200, h=40).pack(pady=8)
        avw = tk.Frame(vc, bg=C['card'])
        avw.pack(fill='x')
        self._analysis_text = styled_text(avw, height=8, color=C['orange'])

        self._refresh_ds()

    # ═════════════════════════════════════════════════════════
    #  PAGE: OUTPUTS
    # ═════════════════════════════════════════════════════════
    def _build_outputs(self):
        sf_frame = ScrollableFrame(self._content)
        self._pages['outputs'] = sf_frame
        f = sf_frame.inner

        section_hdr(f, 'OUTPUT FILES', C['orange'])
        cc = card(f)
        cr = tk.Frame(cc, bg=C['card'])
        cr.pack(fill='x', pady=6)
        NeonBtn(cr, '🔄 REFRESH', cmd=self._refresh_outputs,
                color=C['orange'], w=130, h=36).pack(side='left', padx=(0,8))
        NeonBtn(cr, '📁 OPEN DIR', cmd=lambda: self._open_dir(self.OUTPUT),
                color=C['cyan'], w=130, h=36).pack(side='left', padx=(0,8))
        NeonBtn(cr, '🗑 CLEAR ALL', cmd=self._clear_all_outputs,
                color=C['red'], w=130, h=36).pack(side='left')

        self._out_list_host = tk.Frame(f, bg=C['bg'])
        self._out_list_host.pack(fill='x')

        self._refresh_outputs()

    # ─────────────────────────────────────────────────────────
    #  FOOTER
    # ─────────────────────────────────────────────────────────
    def _build_footer(self):
        ft = tk.Frame(self.root, bg=C['bg2'], height=54)
        ft.pack(fill='x', side='bottom')
        ft.pack_propagate(False)
        tk.Canvas(ft, height=1, bg=C['border'], highlightthickness=0).pack(fill='x')
        row = tk.Frame(ft, bg=C['bg2'])
        row.pack(fill='both', expand=True, padx=16, pady=6)
        self._status_lbl = tk.Label(row, textvariable=self.status_var,
                                    fg=C['cyan'], bg=C['bg2'], font=FONT['sm'])
        self._status_lbl.pack(side='left')
        self._foot_bar = NeonBar(row, w=320, h=10, color=C['cyan'], bg=C['bg2'])
        self._foot_bar.pack(side='left', padx=18)
        tk.Label(row, text='v3.0 PRO', fg=C['dim'], bg=C['bg2'],
                 font=FONT['sm']).pack(side='right')
        if not HAS_LIBROSA:
            tk.Label(row, text='[librosa not found – install for better audio]',
                     fg=C['orange'], bg=C['bg2'], font=FONT['sm']).pack(side='right', padx=12)

    # ═════════════════════════════════════════════════════════
    #  GENERATE LOGIC
    # ═════════════════════════════════════════════════════════
    def _toggle_input(self):
        mode = self.input_type_var.get()
        if mode == 'text':
            self._aud_sec.pack_forget()
            self._txt_sec.pack(fill='x')
        else:
            self._txt_sec.pack_forget()
            self._aud_sec.pack(fill='x')

    def _clear_placeholder(self, _e=None):
        if self._text_box.get('1.0','end-1c') == 'Enter lyrics or speech here...':
            self._text_box.delete('1.0','end')

    def _select_audio(self):
        p = filedialog.askopenfilename(title='Select Audio',
            filetypes=[('Audio','*.wav *.mp3 *.flac *.m4a *.ogg'),('All','*.*')])
        if p:
            self.audio_path_var.set(p)
            self._audio_wave.load(p)
            self._log(f'📂 Loaded: {Path(p).name}')

    def _on_model_select(self, _e=None):
        name = self.model_var.get()
        if not name: return
        # profile
        profile_path = self.MODELS / name / 'voice_profile.json'
        if profile_path.exists():
            with open(profile_path) as f:
                p = json.load(f)
            info = (f"Pitch: {p.get('avg_pitch',0):.0f} Hz  |  "
                    f"Tempo: {p.get('avg_tempo',0):.0f} BPM  |  "
                    f"Files: {p.get('total_files',0)}")
            self._model_info.config(text=info, fg=C['cyan'])
        else:
            persona = self._get_persona(name)
            self._model_info.config(text=persona.get('desc','No profile'), fg=C['gray'])
        # preview waveform
        sample = self._get_sample(name)
        if sample:
            self._prev_wave.load(sample)

    def _get_persona(self, name):
        key = name.lower()
        if key in VOICE_PERSONAS: return VOICE_PERSONAS[key]
        for k in VOICE_PERSONAS:
            if k in key or key in k: return VOICE_PERSONAS[k]
        return VOICE_PERSONAS['default']

    def _get_sample(self, model_name):
        candidates = [
            self.DATASET / model_name,
            self.DATASET / model_name.replace('_custom_voice','').replace('_enhanced_voice',''),
            self.MODELS / model_name / 'samples',
        ]
        for d in candidates:
            if d.exists():
                for ext in ('*.wav','*.mp3','*.flac'):
                    files = list(d.glob(ext))
                    if files: return files[0]
        return None

    def _preview_model(self):
        name = self.model_var.get()
        if not name:
            messagebox.showwarning('No Model', 'Select a model first.'); return
        sample = self._get_sample(name)
        if sample:
            self._log(f'▶ Previewing sample: {sample.name}')
            self._prev_wave.load(sample)
            self._prev_wave.set_active(True)
            threading.Thread(target=self._play_file, args=(str(sample),), daemon=True).start()
        else:
            self._log(f'⚠ No training samples for: {name}')
            messagebox.showinfo('No Samples',
                f"No audio found for '{name}'.\n"
                "Add files via Training tab.")

    def _stop_preview(self):
        self._prev_wave.set_active(False)
        if self._preview_proc:
            try: self._preview_proc.kill()
            except: pass

    def _play_file(self, path):
        try:
            if sys.platform.startswith('linux'):
                self._preview_proc = subprocess.Popen(['xdg-open', path])
            elif sys.platform == 'darwin':
                self._preview_proc = subprocess.Popen(['open', path])
            else:
                self._preview_proc = subprocess.Popen(['start', path], shell=True)
        except Exception as e:
            self._log(f'❌ Playback: {e}')

    def _generate(self):
        if self._generating:
            self._log('⚠ Already generating...'); return
        mode = self.input_type_var.get()
        if mode == 'text':
            txt = self._text_box.get('1.0','end-1c').strip()
            if not txt or txt == 'Enter lyrics or speech here...':
                messagebox.showwarning('No Text','Enter some text.'); return
        else:
            if not self.audio_path_var.get():
                messagebox.showwarning('No Audio','Select an audio file.'); return
        if not self.model_var.get():
            messagebox.showwarning('No Model','Select a voice model.'); return
        threading.Thread(target=self._gen_worker, daemon=True).start()

    def _gen_worker(self):
        self._generating = True
        tmp = None
        try:
            self._prog('🔄 Initializing…', 5)
            model_name = self.model_var.get()
            mode = self.input_type_var.get()

            if mode == 'text':
                txt = self._text_box.get('1.0','end-1c').strip()
                self._prog('🔤 Running TTS…', 20)
                tmp = self._tts(txt)
                self._log(f'📝 TTS done: {len(txt)} chars')
            else:
                tmp = self.audio_path_var.get()
                self._log(f'🎵 Audio: {Path(tmp).name}')

            self._prog('🎵 Loading audio…', 35)
            audio = AudioSegment.from_file(tmp).normalize()

            self._prog('✨ Applying voice profile…', 55)
            persona = self._get_persona(model_name)

            # Load voice profile overrides if available
            vp_path = self.MODELS / model_name / 'voice_profile.json'
            vp = {}
            if vp_path.exists():
                with open(vp_path) as f: vp = json.load(f)

            # FX slider adjustments
            p_adj = self._fx['pitch'].get()
            s_adj = self._fx['speed'].get()
            r_adj = self._fx['reverb'].get()
            b_adj = self._fx['bass'].get()

            persona = dict(persona)
            persona['pitch']  = persona.get('pitch', 0) + p_adj
            persona['speed']  = persona.get('speed', 1.0) * s_adj
            persona['reverb'] = r_adj if r_adj != 0.05 else persona.get('reverb', 0.05)
            persona['gain']   = persona.get('gain', 0) + (b_adj - 1.0) * 4

            audio = self._transform(audio, persona)

            self._prog('💾 Saving…', 85)
            out_dir = self.OUTPUT / model_name
            out_dir.mkdir(exist_ok=True)
            base = 'text_vocal' if mode == 'text' else Path(self.audio_path_var.get()).stem
            ts   = time.strftime('%Y%m%d_%H%M%S')
            name = f'{base}_{model_name}_{ts}.wav'
            out  = out_dir / name
            audio.export(str(out), format='wav')

            if out.exists() and out.stat().st_size > 500:
                self.last_output = out
                self.root.after(0, lambda: self._out_wave.load(out))
                dur  = len(audio) / 1000
                size = out.stat().st_size / 1024
                self._prog('✅ Complete!', 100)
                self._log(f'✅ {name}')
                self._log(f'   Duration: {dur:.1f}s  |  Size: {size:.1f} KB')
                self._log(f'   Saved to: output/{model_name}/')
                self.root.after(0, lambda: self._out_file_lbl.config(
                    text=f'Latest: {name}', fg=C['green']))
                self._refresh_outputs()
                if messagebox.askyesno('✅ Done!',
                        f'{name}\n{dur:.1f}s | {size:.1f}KB\n\nPlay now?'):
                    self._play_output()
            else:
                raise RuntimeError('Output file empty or missing')

        except Exception as e:
            self._prog('❌ Error', 0)
            self._log(f'❌ {e}')
            messagebox.showerror('Error', str(e))
        finally:
            if tmp and mode == 'text' and os.path.exists(tmp):
                try: os.unlink(tmp)
                except: pass
            self._generating = False

    def _tts(self, text):
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name); mp3 = f.name
        seg = AudioSegment.from_mp3(mp3)
        os.unlink(mp3)
        wav = mp3.replace('.mp3','.wav')
        seg.export(wav, format='wav', parameters=['-ar','44100'])
        return wav

    def _transform(self, audio, persona):
        """Apply voice transformation – uses librosa when available."""
        pitch  = persona.get('pitch', 0)
        speed  = persona.get('speed', 1.0)
        reverb = persona.get('reverb', 0.0)
        gain   = persona.get('gain', 0)

        if HAS_LIBROSA and (pitch != 0 or speed != 1.0):
            # Convert to numpy → librosa → back
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples /= (2**(audio.sample_width*8 - 1))
            sr = audio.frame_rate
            if audio.channels > 1:
                samples = samples.reshape(-1, audio.channels).mean(axis=1)
            if pitch != 0:
                samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=float(pitch))
            if speed != 1.0:
                samples = librosa.effects.time_stretch(samples, rate=float(speed))
            # Back to pydub
            samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            audio = AudioSegment(
                samples.tobytes(), frame_rate=sr,
                sample_width=2, channels=1)
        else:
            # Fallback: frame-rate pitch trick
            if pitch != 0:
                factor  = 2 ** (pitch / 12.0)
                new_rate = int(audio.frame_rate * factor)
                audio   = audio._spawn(audio.raw_data, overrides={'frame_rate': new_rate})
                audio   = audio.set_frame_rate(44100)
            if speed != 1.0:
                audio = audio.speedup(playback_speed=speed)

        # Gain / EQ
        if gain != 0:
            audio = audio + gain

        # Reverb (echo overlay)
        if reverb > 0.02:
            delay = int(70 * reverb)
            echo  = audio - int(5 + reverb * 5)
            try:   audio = audio.overlay(echo, position=delay)
            except: pass

        return audio

    def _play_output(self):
        if not self.last_output or not self.last_output.exists():
            messagebox.showinfo('No Output','Generate audio first.'); return
        self._out_wave.set_active(True)
        threading.Thread(target=self._play_file,
                         args=(str(self.last_output),), daemon=True).start()
        threading.Thread(target=self._vu_animate, daemon=True).start()

    def _vu_animate(self):
        for _ in range(180):
            self._vu.push(random.uniform(0.25, 0.85))
            time.sleep(0.05)
        self._out_wave.set_active(False)

    def _open_out_folder(self):
        out = self.OUTPUT / (self.model_var.get() or '')
        self._open_dir(out if out.exists() else self.OUTPUT)

    # ═════════════════════════════════════════════════════════
    #  MODELS LOGIC
    # ═════════════════════════════════════════════════════════
    def _refresh_models(self):
        models = []
        for item in sorted(self.MODELS.iterdir()):
            if item.is_file() and item.suffix == '.pth':
                models.append(item.stem)
            elif item.is_dir():
                for sub in item.iterdir():
                    if sub.suffix == '.pth':
                        models.append(item.name); break
        self._gen_model_cb['values'] = models
        if hasattr(self, '_train_cb'):
            self._train_cb['values'] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])
            self._on_model_select()
        self._rebuild_model_cards(models)
        self._log(f'🔍 {len(models)} model(s) found')
        self._prog(f'✅ {len(models)} models loaded', 100)

    def _rebuild_model_cards(self, models):
        for w in self._model_cards_host.winfo_children():
            w.destroy()
        if not models:
            tk.Label(self._model_cards_host,
                     text='No models. Import a .pth file.',
                     fg=C['gray'], bg=C['bg'], font=FONT['body']).pack(pady=20)
            return
        for name in models:
            self._model_card(self._model_cards_host, name)

    def _model_card(self, parent, name):
        frame = tk.Frame(parent, bg=C['card'],
                         highlightthickness=1, highlightbackground=C['border'])
        frame.pack(fill='x', padx=12, pady=4)
        inner = tk.Frame(frame, bg=C['card'])
        inner.pack(fill='x', padx=14, pady=10)

        info = tk.Frame(inner, bg=C['card'])
        info.pack(side='left', fill='x', expand=True)
        tk.Label(info, text=name, fg=C['purple'], bg=C['card'],
                 font=FONT['h2']).pack(anchor='w')

        # sample count
        ds_dir = self.DATASET / name
        n = sum(len(list(ds_dir.glob(f'*{e}')))
                for e in ('.wav','.mp3','.flac')) if ds_dir.exists() else 0
        sample = self._get_sample(name)
        meta = f'Training clips: {n}'
        if sample: meta += f'   Preview: {sample.name}'
        tk.Label(info, text=meta, fg=C['gray'], bg=C['card'],
                 font=FONT['sm']).pack(anchor='w')

        btns = tk.Frame(inner, bg=C['card'])
        btns.pack(side='right')
        NeonBtn(btns, '▶ PREVIEW', cmd=lambda n=name: self._preview_by_name(n),
                color=C['green'], w=110, h=32).pack(side='left', padx=(0,6))
        NeonBtn(btns, 'SELECT', cmd=lambda n=name: self._select_model(n),
                color=C['purple'], w=90, h=32).pack(side='left', padx=(0,6))
        NeonBtn(btns, '🗑', cmd=lambda n=name: self._delete_model(n),
                color=C['red'], w=38, h=32).pack(side='left')

    def _preview_by_name(self, name):
        self.model_var.set(name)
        self._preview_model()

    def _select_model(self, name):
        self.model_var.set(name)
        self._on_model_select()
        self._show('generate')
        self._log(f'✅ Model selected: {name}')

    def _delete_model(self, name):
        if messagebox.askyesno('Delete', f"Delete model '{name}'?"):
            for p in [self.MODELS/name, self.MODELS/f'{name}.pth']:
                if p.is_dir():  shutil.rmtree(p)
                elif p.is_file(): p.unlink()
            self._refresh_models()
            self._log(f'🗑 Deleted: {name}')

    def _import_model_file(self):
        p = filedialog.askopenfilename(
            title='Import Model', filetypes=[('PTH','*.pth'),('All','*.*')])
        if p:
            shutil.copy2(p, self.MODELS/Path(p).name)
            self._refresh_models()
            self._log(f'📥 Imported: {Path(p).name}')

    def _import_model_folder(self):
        p = filedialog.askdirectory(title='Import Model Folder')
        if p:
            src = Path(p)
            shutil.copytree(str(src), str(self.MODELS/src.name), dirs_exist_ok=True)
            self._refresh_models()
            self._log(f'📥 Imported folder: {src.name}')

    # ═════════════════════════════════════════════════════════
    #  TRAINING LOGIC
    # ═════════════════════════════════════════════════════════
    def _new_model_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title('New Model')
        dlg.configure(bg=C['bg2'])
        dlg.geometry('380x180')
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text='MODEL NAME', fg=C['cyan'], bg=C['bg2'],
                 font=FONT['h2']).pack(pady=(18,6))
        nv = tk.StringVar()
        e = tk.Entry(dlg, textvariable=nv, bg=C['bg3'], fg=C['white'],
                     insertbackground=C['cyan'], font=FONT['body'],
                     relief='flat', width=28)
        e.pack(ipady=8, pady=6)
        e.focus_set()
        def create():
            name = nv.get().strip().replace(' ','_')
            if name:
                (self.MODELS/name).mkdir(exist_ok=True)
                (self.DATASET/name).mkdir(exist_ok=True)
                self._refresh_models()
                self.train_model_var.set(name)
                self._refresh_ds()
                self._log(f'✅ Created: {name}')
                dlg.destroy()
        NeonBtn(dlg, 'CREATE', cmd=create, color=C['green'], w=140, h=38).pack(pady=8)
        e.bind('<Return>', lambda _: create())

    def _import_clip(self):
        m = self.train_model_var.get()
        if not m: messagebox.showwarning('No Model','Select target model.'); return
        p = filedialog.askopenfilename(
            title='Select Audio',
            filetypes=[('Audio','*.wav *.mp3 *.flac *.m4a *.ogg'),('All','*.*')])
        if p:
            d = self.DATASET/m; d.mkdir(exist_ok=True)
            shutil.copy2(p, d/Path(p).name)
            self._log(f'📥 Imported → {m}/: {Path(p).name}')
            self._refresh_ds()

    def _import_ds_folder(self):
        m = self.train_model_var.get()
        if not m: messagebox.showwarning('No Model','Select target model.'); return
        folder = filedialog.askdirectory(title='Select Audio Folder')
        if folder:
            d = self.DATASET/m; d.mkdir(exist_ok=True)
            count = 0
            for ext in ('*.wav','*.mp3','*.flac','*.m4a','*.ogg'):
                for fp in Path(folder).rglob(ext):
                    dest = d/fp.name
                    if not dest.exists():
                        shutil.copy2(fp, dest); count += 1
            self._log(f'📥 {count} files → {m}/')
            self._refresh_ds()

    def _open_ds_folder(self):
        m = self.train_model_var.get()
        target = (self.DATASET/m) if m else self.DATASET
        target.mkdir(exist_ok=True)
        self._open_dir(target)

    def _refresh_ds(self, _e=None):
        self._ds_text.delete('1.0','end')
        m = self.train_model_var.get()
        dirs = [self.DATASET/m] if m else list(self.DATASET.iterdir())
        total = 0
        for d in dirs:
            if not d.is_dir(): continue
            files = sorted([f for ext in ('.wav','.mp3','.flac','.m4a','.ogg')
                             for f in d.glob(f'*{ext}')])
            if not files: continue
            self._ds_text.insert('end', f'📁 {d.name}/   ({len(files)} files)\n')
            for fp in files[:30]:
                self._ds_text.insert('end', f'   ├ {fp.name}  ({fp.stat().st_size/1024:.1f}KB)\n')
            if len(files) > 30:
                self._ds_text.insert('end', f'   └ …{len(files)-30} more\n')
            self._ds_text.insert('end', '\n')
            total += len(files)
        if total == 0:
            self._ds_text.insert('end', 'No audio files yet. Import some!\n')

    def _analyze_voice(self):
        if not HAS_LIBROSA:
            messagebox.showwarning('Missing','pip install librosa'); return
        m = self.train_model_var.get()
        if not m: messagebox.showwarning('No Model','Select a model.'); return
        threading.Thread(target=self._analyze_worker, args=(m,), daemon=True).start()

    def _analyze_worker(self, name):
        self._prog('🔬 Analyzing…', 10)
        d = self.DATASET / name
        if not d.exists():
            self._log(f'❌ Dataset not found: {d}'); return
        files = [f for ext in ('*.wav','*.mp3','*.flac') for f in d.glob(ext)]
        if not files:
            self._log('❌ No audio files'); return
        pitches, tempos, energies = [], [], []
        for i, fp in enumerate(files[:10]):
            try:
                y, sr = librosa.load(str(fp), sr=22050, duration=30)
                pa, _ = librosa.piptrack(y=y, sr=sr)
                p = pa[pa>50]
                if len(p): pitches.append(float(np.mean(p)))
                t, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempos.append(float(t))
                energies.append(float(np.mean(librosa.feature.rms(y=y)[0])))
                self._prog(f'Analyzing {i+1}/{min(10,len(files))}…', 10 + (i+1)*8)
            except Exception as e:
                self._log(f'⚠ {fp.name}: {e}')
        if pitches:
            profile = {
                'speaker': name, 'total_files': len(files),
                'avg_pitch': float(np.mean(pitches)),
                'pitch_range': [float(np.min(pitches)), float(np.max(pitches))],
                'avg_tempo':  float(np.mean(tempos)),
                'avg_energy': float(np.mean(energies)),
            }
            out = self.MODELS/name
            out.mkdir(exist_ok=True)
            with open(out/'voice_profile.json','w') as f:
                json.dump(profile, f, indent=2)
            txt = (f'🎤 VOICE PROFILE: {name}\n{"="*38}\n'
                   f'Avg Pitch : {profile["avg_pitch"]:.1f} Hz\n'
                   f'Range     : {profile["pitch_range"][0]:.0f}–{profile["pitch_range"][1]:.0f} Hz\n'
                   f'Avg Tempo : {profile["avg_tempo"]:.1f} BPM\n'
                   f'Energy    : {profile["avg_energy"]:.5f}\n'
                   f'Files     : {len(files)}\n'
                   f'\nSaved → models/{name}/voice_profile.json\n')
            def _show():
                self._analysis_text.delete('1.0','end')
                self._analysis_text.insert('1.0', txt)
            self.root.after(0, _show)
            self._log(f'✅ Analysis complete: {name}')
        self._prog('✅ Done', 100)

    # ═════════════════════════════════════════════════════════
    #  OUTPUTS LOGIC
    # ═════════════════════════════════════════════════════════
    def _refresh_outputs(self):
        for w in self._out_list_host.winfo_children():
            w.destroy()
        all_files = []
        if self.OUTPUT.exists():
            for d in sorted(self.OUTPUT.iterdir()):
                if d.is_dir():
                    for fp in sorted(d.glob('*.wav'), reverse=True):
                        all_files.append((d.name, fp))
                elif d.suffix == '.wav':
                    all_files.append(('root', d))
        if not all_files:
            tk.Label(self._out_list_host, text='No outputs yet.',
                     fg=C['gray'], bg=C['bg'], font=FONT['body']).pack(pady=20)
            return
        cur = None
        for model, fp in all_files[:60]:
            if model != cur:
                cur = model
                section_hdr(self._out_list_host, f'📁 {model}/', C['orange'], C['bg'])
            self._out_row(self._out_list_host, fp)

    def _out_row(self, parent, fp):
        row = tk.Frame(parent, bg=C['card'],
                       highlightthickness=1, highlightbackground=C['border'])
        row.pack(fill='x', padx=12, pady=2)
        inner = tk.Frame(row, bg=C['card'])
        inner.pack(fill='x', padx=12, pady=6)
        sz = fp.stat().st_size / 1024
        tk.Label(inner, text=fp.name, fg=C['white'], bg=C['card'],
                 font=FONT['sm']).pack(side='left')
        tk.Label(inner, text=f'{sz:.1f}KB', fg=C['gray'], bg=C['card'],
                 font=FONT['sm']).pack(side='left', padx=(8,0))
        NeonBtn(inner, '▶', cmd=lambda f=fp: self._play_file(str(f)),
                color=C['green'], w=38, h=28).pack(side='right', padx=(4,0))
        NeonBtn(inner, '🗑', cmd=lambda f=fp: self._del_output(f),
                color=C['red'], w=38, h=28).pack(side='right', padx=(4,0))

    def _del_output(self, fp):
        if messagebox.askyesno('Delete', f'Delete {fp.name}?'):
            fp.unlink()
            self._refresh_outputs()

    def _clear_all_outputs(self):
        if messagebox.askyesno('Clear All', 'Delete ALL output files? This cannot be undone.'):
            for item in self.OUTPUT.iterdir():
                if item.is_dir():  shutil.rmtree(item)
                elif item.suffix == '.wav': item.unlink()
            self._refresh_outputs()
            self._log('🗑 All outputs cleared')

    # ─────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────
    def _log(self, msg):
        def _do():
            ts = time.strftime('%H:%M:%S')
            self._out_log.insert('end', f'[{ts}] {msg}\n')
            self._out_log.see('end')
        self.root.after(0, _do)

    def _prog(self, text, value):
        def _do():
            self.status_var.set(text)
            self._foot_bar.set(value)
        self.root.after(0, _do)

    def _open_dir(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        try:
            if sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', str(path)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(path)])
            else:
                os.startfile(str(path))
        except Exception as e:
            self._log(f'❌ {e}')

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.configure(bg=C['bg'])
    app = StudioPro(root)
    root.mainloop()

if __name__ == '__main__':
    main()
