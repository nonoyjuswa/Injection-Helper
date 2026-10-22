#!/usr/bin/env python3
"""PrivARcy Mock-up Injection Helper

A small Windows-friendly Tkinter utility for creating synthetic mock-up composites.
Dependencies: pillow, opencv-python, numpy
Install: py -m pip install pillow opencv-python numpy
Run: py privarcy_injection_helper.py

Workflow:
1) Mark each background image: click tabletop corners clockwise starting at far-left,
   then far-right, near-right, near-left. Add exclusion polygons for objects to avoid.
2) Save profile. Repeat for each background.
3) Generate composites from transparent PNG mock-ups using saved profiles.

This tool is a dataset-assistance utility. Inspect generated images before training.
"""
from __future__ import annotations
import json, random, traceback, math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import cv2

APP_TITLE = "PrivARcy Mock-up Injection Helper"
PROFILE_DIRNAME = "profiles"

# ---- Default folder paths (edit these if your folders change) ----
DEFAULT_BG_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Backgrounds"
DEFAULT_MOCK_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Mockups"
DEFAULT_OUT_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\Automated Injection\Output"
DEFAULT_PROFILE_DIR = r"C:\Users\Juswa\Documents\Thesis\DATASETS\helper\profiles"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def point_in_poly(point, polygon):
    x, y = point; inside = False
    if len(polygon) < 3: return False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]; xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi) * (y-yi) / ((yj-yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def segments_intersect(a,b,c,d):
    def orient(p,q,r): return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    o1,o2,o3,o4=orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
    return (o1 == 0 or o2 == 0 or (o1 < 0) != (o2 < 0)) and (o3 == 0 or o4 == 0 or (o3 < 0) != (o4 < 0))


def polygons_overlap(poly1, poly2):
    if any(point_in_poly(p, poly2) for p in poly1): return True
    if any(point_in_poly(p, poly1) for p in poly2): return True
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            if segments_intersect(poly1[i], poly1[(i+1)%len(poly1)], poly2[j], poly2[(j+1)%len(poly2)]): return True
    return False


def ensure_rgba(path):
    """Load mock-up and trim transparent padding so canvas whitespace does not change its apparent size."""
    im = Image.open(path).convert('RGBA')
    alpha = im.getchannel('A')
    bbox = alpha.getbbox()
    if bbox is None: raise ValueError('Mock-up image is empty/fully transparent.')
    return im.crop(bbox)


def warp_rgba(src_rgba, dst_quad, out_size):
    """Warp RGBA image corners to destination quadrilateral in output-image coordinates."""
    sw, sh = src_rgba.size
    src = np.float32([[0,0],[sw-1,0],[sw-1,sh-1],[0,sh-1]])
    dst = np.float32(dst_quad)
    matrix = cv2.getPerspectiveTransform(src, dst)
    arr = np.array(src_rgba)
    warped = cv2.warpPerspective(arr, matrix, out_size, flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return Image.fromarray(warped, 'RGBA')


class MarkingWindow(tk.Toplevel):
    def __init__(self, app, image_path):
        super().__init__(app)
        self.app=app; self.image_path=Path(image_path)
        self.title('Mark tabletop and exclusion zones - ' + self.image_path.name)
        self.geometry('1100x780'); self.minsize(720,520)
        self.original=Image.open(self.image_path).convert('RGB')
        self.W,self.H=self.original.size
        self.table=[]; self.obstacles=[]; self.current=[]; self.mode='table'; self.profile_path=None
        self.mockup_calibrations={}; self.calibration_name=tk.StringVar(value='card'); self.size_quad=[]; self.drag_size_index=None
        existing=app.load_profile(self.image_path)
        if existing:
            self.table=[tuple(p) for p in existing.get('table',[])]
            self.obstacles=[[tuple(p) for p in poly] for poly in existing.get('obstacles',[])]
            self.mockup_calibrations={k:[tuple(p) for p in quad] for k,quad in existing.get('mockup_calibrations',{}).items()}
        self.canvas=tk.Canvas(self, bg='#222', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        controls=ttk.Frame(self, padding=8); controls.pack(fill='x')
        ttk.Button(controls,text='Mark tabletop (4 corners)',command=self.set_table).pack(side='left',padx=3)
        ttk.Button(controls,text='Add obstacle polygon',command=self.set_obstacle).pack(side='left',padx=3)
        ttk.Button(controls,text='Finish obstacle',command=self.finish_obstacle).pack(side='left',padx=3)
        ttk.Button(controls,text='Undo last point',command=self.undo).pack(side='left',padx=3)
        ttk.Button(controls,text='Clear current',command=self.clear_current).pack(side='left',padx=3)
        ttk.Label(controls,text='Mock-up class/name:').pack(side='left',padx=(8,2))
        ttk.Entry(controls,textvariable=self.calibration_name,width=12).pack(side='left')
        ttk.Button(controls,text='Set/edit sample size (4 points)',command=self.set_size_calibration).pack(side='left',padx=3)
        ttk.Button(controls,text='Save profile',command=self.save).pack(side='right',padx=3)
        self.status=ttk.Label(self,text='Click four tabletop corners in order: far-left, far-right, near-right, near-left.',padding=6)
        self.status.pack(fill='x')
        self.canvas.bind('<Button-1>',self.click); self.canvas.bind('<B1-Motion>',self.drag_size_point); self.canvas.bind('<ButtonRelease-1>',self.release_size_point); self.canvas.bind('<Configure>',lambda e:self.redraw())
        self.set_table(); self.after(100,self.redraw)

    def set_table(self):
        self.mode='table'; self.current=[]
        self.status.config(text='Tabletop: click 4 corners clockwise: far-left, far-right, near-right, near-left.')
        self.redraw()
    def set_size_calibration(self):
        key=self.calibration_name.get().strip().lower()
        if not key:
            messagebox.showwarning('Name required','Enter a mock-up class/name (for example: id_card, bank_card, passport).',parent=self); return
        self.mode='size'; self.current=[]; self.drag_size_index=None
        self.size_quad=[tuple(p) for p in self.mockup_calibrations.get(key,[])]
        self.status.config(text=f'Sample size for “{key}”: click 4 corners clockwise around the intended mock-up footprint. Drag the green points to fine-tune them. Save profile when finished.')
        self.redraw()
    def drag_size_point(self,e):
        if self.mode!='size' or self.drag_size_index is None or len(self.size_quad)!=4: return
        self.size_quad[self.drag_size_index]=self.to_image(e.x,e.y)
        self.redraw()
    def release_size_point(self,e):
        if self.mode=='size' and len(self.size_quad)==4:
            key=self.calibration_name.get().strip().lower()
            if key: self.mockup_calibrations[key]=[tuple(q) for q in self.size_quad]
        self.drag_size_index=None
    def set_obstacle(self):
        if len(self.table)!=4:
            messagebox.showwarning('Tabletop needed','Mark the 4 tabletop corners first.',parent=self); return
        self.mode='obstacle'; self.current=[]
        self.status.config(text='Obstacle: click at least 3 points around an object to avoid, then click Finish obstacle.')
        self.redraw()
    def finish_obstacle(self):
        if self.mode!='obstacle' or len(self.current)<3:
            messagebox.showinfo('More points needed','An obstacle polygon needs at least 3 points.',parent=self); return
        self.obstacles.append(self.current[:]); self.current=[]
        self.status.config(text=f'Saved obstacle polygon ({len(self.obstacles)} total). Add another obstacle or save profile.')
        self.redraw()
    def clear_current(self): self.current=[]; self.redraw()
    def undo(self):
        if self.current: self.current.pop()
        elif self.mode=='obstacle' and self.obstacles: self.obstacles.pop()
        elif self.mode=='table' and self.table: self.table.pop()
        self.redraw()
    def scale(self):
        cw=max(self.canvas.winfo_width(),100); ch=max(self.canvas.winfo_height(),100)
        s=min((cw-20)/self.W,(ch-20)/self.H)
        return s,(cw-self.W*s)/2,(ch-self.H*s)/2
    def to_image(self,x,y):
        s,ox,oy=self.scale(); return (max(0,min(self.W-1,(x-ox)/s)),max(0,min(self.H-1,(y-oy)/s)))
    def to_canvas(self,p):
        s,ox,oy=self.scale(); return p[0]*s+ox,p[1]*s+oy
    def click(self,e):
        p=self.to_image(e.x,e.y)
        if self.mode=='size':
            if len(self.size_quad)==4:
                # Start dragging the closest existing size corner, if clicked nearby.
                distances=[math.hypot(self.to_canvas(q)[0]-e.x,self.to_canvas(q)[1]-e.y) for q in self.size_quad]
                i=min(range(4),key=distances.__getitem__)
                if distances[i] <= 18: self.drag_size_index=i
                else: self.size_quad=[]; self.size_quad.append(p)
            else:
                self.size_quad.append(p)
            if len(self.size_quad)==4:
                key=self.calibration_name.get().strip().lower()
                self.mockup_calibrations[key]=[tuple(q) for q in self.size_quad]
                self.status.config(text=f'Sample size set for “{key}”. Drag the green corner points to adjust; Save profile to keep it.')
        elif self.mode=='table':
            if len(self.table)<4: self.table.append(p)
            else: self.table=[p]
            if len(self.table)==4:
                self.mode='obstacle'; self.status.config(text='Tabletop corners marked. Click Add obstacle polygon to mark objects to avoid, or Save profile.')
        else: self.current.append(p)
        self.redraw()
    def redraw(self):
        if not hasattr(self,'canvas'): return
        self.canvas.delete('all')
        cw=max(self.canvas.winfo_width(),100); ch=max(self.canvas.winfo_height(),100)
        s,ox,oy=self.scale()
        preview=self.original.copy(); preview.thumbnail((int(cw-20),int(ch-20)),Image.Resampling.LANCZOS)
        # Fit uses the same geometry as scale; resize exactly for coordinate consistency.
        preview=self.original.resize((max(1,int(self.W*s)),max(1,int(self.H*s))),Image.Resampling.LANCZOS)
        self.tkimg=ImageTk.PhotoImage(preview)
        self.canvas.create_image(ox,oy,anchor='nw',image=self.tkimg)
        if len(self.table)>=2:
            pts=[self.to_canvas(p) for p in self.table]
            if len(pts)>=3: self.canvas.create_polygon(*[v for p in pts for v in p],outline='#2bff8a',fill='',width=2)
            for i,p in enumerate(pts):
                self.canvas.create_oval(p[0]-5,p[1]-5,p[0]+5,p[1]+5,fill='#2bff8a')
                self.canvas.create_text(p[0]+10,p[1]-10,text=str(i+1),fill='white')
        if self.size_quad:
            pts=[self.to_canvas(p) for p in self.size_quad]
            if len(pts)>=2:
                self.canvas.create_line(*[v for p in pts for v in p],fill='#4de1ff',width=3,joinstyle='round')
                if len(pts)==4: self.canvas.create_line(*self.to_canvas(self.size_quad[0]),*self.to_canvas(self.size_quad[-1]),fill='#4de1ff',width=3)
            for i,p in enumerate(pts):
                self.canvas.create_oval(p[0]-7,p[1]-7,p[0]+7,p[1]+7,fill='#4de1ff',outline='white',width=1)
                self.canvas.create_text(p[0]+11,p[1]-11,text=f'S{i+1}',fill='#4de1ff')
        for poly in self.obstacles:
            pts=[self.to_canvas(p) for p in poly]
            self.canvas.create_polygon(*[v for p in pts for v in p],outline='#ff5757',fill='#ff5757',stipple='gray25',width=2)
        if self.current:
            pts=[self.to_canvas(p) for p in self.current]
            if len(pts)>1: self.canvas.create_line(*[v for p in pts for v in p],fill='#ffd34e',width=2)
            for p in pts: self.canvas.create_oval(p[0]-4,p[1]-4,p[0]+4,p[1]+4,fill='#ffd34e')
    def save(self):
        if len(self.table)!=4:
            messagebox.showerror('Incomplete profile','Please mark exactly four tabletop corners.',parent=self); return
        try:
            self.app.save_profile(self.image_path, {'image':str(self.image_path.resolve()),'size':[self.W,self.H],
                'table':[list(p) for p in self.table], 'obstacles':[[list(p) for p in poly] for poly in self.obstacles],
                'mockup_calibrations':{k:[list(p) for p in quad] for k,quad in self.mockup_calibrations.items()}})
            messagebox.showinfo('Saved','Background profile saved.',parent=self)
        except Exception as e: self.app.log_error('save profile',self.image_path,e); messagebox.showerror('Save failed',str(e),parent=self)


class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title(APP_TITLE); self.geometry('800x620'); self.minsize(700,560)
        self.bg_dir=tk.StringVar(value=DEFAULT_BG_DIR); self.mock_dir=tk.StringVar(value=DEFAULT_MOCK_DIR); self.out_dir=tk.StringVar(value=DEFAULT_OUT_DIR); self.profile_dir=tk.StringVar(value=DEFAULT_PROFILE_DIR)
        self.count=tk.IntVar(value=3); self.seed=tk.StringVar(value=''); self.min_scale=tk.DoubleVar(value=0.20); self.max_scale=tk.DoubleVar(value=0.20); self.objects_per_image=tk.IntVar(value=1); self.max_rotation=tk.IntVar(value=25)
        self.log_path=Path.cwd()/'privarcy_injection_errors.log'
        self.build_ui()
    def build_ui(self):
        root=ttk.Frame(self,padding=16); root.pack(fill='both',expand=True)
        ttk.Label(root,text='PrivARcy Mock-up Injection Helper',font=('Segoe UI',16,'bold')).pack(anchor='w')
        ttk.Label(root,text='Mark each tabletop once, define objects to avoid, set consistent mock-up sizing, and generate composites with multiple mock-ups per image.',wraplength=740).pack(anchor='w',pady=(3,14))
        group=ttk.LabelFrame(root,text='1. Mark a background',padding=10); group.pack(fill='x',pady=5)
        ttk.Label(group,text='Background image:').grid(row=0,column=0,sticky='w'); self.mark_path=tk.StringVar()
        ttk.Entry(group,textvariable=self.mark_path).grid(row=0,column=1,sticky='ew',padx=6)
        ttk.Button(group,text='Browse…',command=self.pick_mark).grid(row=0,column=2)
        ttk.Button(group,text='Open marking window',command=self.open_marking).grid(row=1,column=1,sticky='w',pady=8)
        group.columnconfigure(1,weight=1)
        group=ttk.LabelFrame(root,text='2. Generate composites',padding=10); group.pack(fill='x',pady=5)
        self.path_row(group,0,'Background folder',self.bg_dir,lambda:self.pick_dir(self.bg_dir))
        self.path_row(group,1,'Transparent mock-up PNG folder',self.mock_dir,lambda:self.pick_dir(self.mock_dir))
        self.path_row(group,2,'Profiles folder',self.profile_dir,lambda:self.pick_dir(self.profile_dir))
        self.path_row(group,3,'Output folder',self.out_dir,lambda:self.pick_dir(self.out_dir))
        row=ttk.Frame(group); row.grid(row=4,column=0,columnspan=3,sticky='ew',pady=(10,0))
        ttk.Label(row,text='Composites per mock-up').pack(side='left'); ttk.Spinbox(row,from_=1,to=1000,textvariable=self.count,width=7).pack(side='left',padx=(5,16))
        ttk.Label(row,text='Mock-up width fraction').pack(side='left'); ttk.Label(row,text='min').pack(side='left',padx=(8,2)); ttk.Entry(row,textvariable=self.min_scale,width=6).pack(side='left')
        ttk.Label(row,text='max').pack(side='left',padx=(8,2)); ttk.Entry(row,textvariable=self.max_scale,width=6).pack(side='left')
        row=ttk.Frame(group); row.grid(row=5,column=0,columnspan=3,sticky='ew',pady=(8,2))
        ttk.Label(row,text='Mock-ups per image').pack(side='left'); ttk.Spinbox(row,from_=1,to=20,textvariable=self.objects_per_image,width=5).pack(side='left',padx=(5,16))
        ttk.Label(row,text='Random rotation ± degrees').pack(side='left'); ttk.Spinbox(row,from_=0,to=180,textvariable=self.max_rotation,width=5).pack(side='left',padx=5)
        row=ttk.Frame(group); row.grid(row=6,column=0,columnspan=3,sticky='ew',pady=8)
        ttk.Label(row,text='Random seed (optional)').pack(side='left'); ttk.Entry(row,textvariable=self.seed,width=12).pack(side='left',padx=6)
        ttk.Button(row,text='Generate composites',command=self.generate).pack(side='right')
        self.progress=ttk.Progressbar(root,mode='determinate'); self.progress.pack(fill='x',pady=(12,4))
        self.status=ttk.Label(root,text='Ready. Recommended: use transparent PNG mock-ups and review every output.'); self.status.pack(anchor='w')
        ttk.Button(root,text='Open error log',command=self.open_log).pack(anchor='e',pady=(6,0))
        ttk.Label(root,text='Profiles are stored as JSON and match backgrounds by filename. Avoid duplicate background filenames.',foreground='#555',wraplength=740).pack(anchor='w',pady=(8,0))
    def path_row(self,parent,r,label,var,cmd):
        ttk.Label(parent,text=label+':').grid(row=r,column=0,sticky='w',pady=3)
        ttk.Entry(parent,textvariable=var).grid(row=r,column=1,sticky='ew',padx=6,pady=3)
        ttk.Button(parent,text='Browse…',command=cmd).grid(row=r,column=2,pady=3)
        parent.columnconfigure(1,weight=1)
    def pick_dir(self,var):
        p=filedialog.askdirectory(parent=self,initialdir=var.get() if Path(var.get()).is_dir() else None)
        if p: var.set(p)
    def pick_mark(self):
        p=filedialog.askopenfilename(parent=self,initialdir=self.bg_dir.get() if Path(self.bg_dir.get()).is_dir() else None,filetypes=[('Images','*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff'),('All files','*.*')])
        if p: self.mark_path.set(p)
    def open_marking(self):
        p=self.mark_path.get().strip()
        if not p or not Path(p).is_file(): messagebox.showerror('Choose image','Select an existing background image first.'); return
        try: MarkingWindow(self,p)
        except Exception as e: self.log_error('open marking window',p,e); messagebox.showerror('Could not open image',f'{e}\n\nSee log: {self.log_path}')
    def profile_path(self,image_path): return Path(self.profile_dir.get())/(Path(image_path).stem+'.json')
    def load_profile(self,image_path):
        p=self.profile_path(image_path)
        if not p.is_file(): return None
        try:
            data=json.loads(p.read_text(encoding='utf-8'))
            # Scale coordinates if the background image dimensions changed.
            im=Image.open(image_path); w,h=im.size
            old=data.get('size',[w,h]); sx=w/old[0]; sy=h/old[1]
            data['table']=[[x*sx,y*sy] for x,y in data.get('table',[])]
            data['obstacles']=[[[x*sx,y*sy] for x,y in poly] for poly in data.get('obstacles',[])]
            data['mockup_calibrations']={k:[[x*sx,y*sy] for x,y in quad] for k,quad in data.get('mockup_calibrations',{}).items()}
            return data
        except Exception as e: self.log_error('load profile',image_path,e); return None
    def save_profile(self,image_path,data):
        p=self.profile_path(image_path); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(data,indent=2),encoding='utf-8')
    def log_error(self,task,item,error):
        try:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a',encoding='utf-8') as f:
                f.write(f'\n[{__import__("datetime").datetime.now().isoformat()}] {task}: {item}\n{error}\n{traceback.format_exc()}\n')
        except Exception: pass
    def open_log(self):
        if not self.log_path.exists(): self.log_path.write_text('No errors logged yet.\n',encoding='utf-8')
        try:
            import os,sys,subprocess
            if sys.platform.startswith('win'): os.startfile(str(self.log_path))
            else: subprocess.Popen(['xdg-open',str(self.log_path)])
        except Exception as e: messagebox.showinfo('Log location',str(self.log_path))
    def generate(self):
        try:
            bgroot=Path(self.bg_dir.get()); mockroot=Path(self.mock_dir.get()); outroot=Path(self.out_dir.get())
            if not bgroot.is_dir() or not mockroot.is_dir() or not self.out_dir.get().strip(): raise ValueError('Choose valid background, mock-up, and output folders.')
            backgrounds=sorted(p for p in bgroot.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            mockups=sorted(p for p in mockroot.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if not backgrounds: raise ValueError('No supported images found in the background folder.')
            if not mockups: raise ValueError('No supported images found in the mock-up folder.')
            n=int(self.count.get()); object_count=int(self.objects_per_image.get()); max_rotation=int(self.max_rotation.get()); lo=float(self.min_scale.get()); hi=float(self.max_scale.get())
            if n<1 or object_count<1 or object_count>20 or max_rotation<0 or max_rotation>180 or lo<=0 or hi<lo or hi>0.9: raise ValueError('Check settings: composites and mock-ups per image must be positive; width fractions must satisfy 0 < min <= max <= 0.9.')
            if object_count > len(mockups): raise ValueError(f'You selected {object_count} mock-ups per image, but only {len(mockups)} mock-up files are available. Add more files or lower the setting.')
            rng=random.Random(int(self.seed.get()) if self.seed.get().strip() else None)
            outroot.mkdir(parents=True,exist_ok=True); total=len(mockups)*n; self.progress['maximum']=total; self.progress['value']=0
            success=fail=0; errors=[]
            profiles={}
            for bg in backgrounds:
                prof=self.load_profile(bg)
                if prof and len(prof.get('table',[]))==4: profiles[bg]=prof
            if not profiles: raise ValueError('No valid saved profiles found. Mark backgrounds first, and ensure the Profiles folder matches the marking window.')

            def tabletop_homography(table):
                """Return the exact projective mapping from a flat unit tabletop to image pixels.

                The four marked points are ordered far-left, far-right, near-right,
                near-left. Unlike bilinear interpolation, a homography models the same
                planar perspective transform used by perspective-warp tools such as
                Photoshop's four-corner transform.
                """
                plane = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
                image = np.float32(table)
                H = cv2.getPerspectiveTransform(plane, image)
                if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
                    raise ValueError('The four tabletop points do not define a valid perspective plane. Mark the corners clockwise: far-left, far-right, near-right, near-left.')
                return H

            def project_table_point(Htable, u, v):
                """Project a point from tabletop-plane coordinates into the original image."""
                pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, Htable)[0, 0]
                return [float(mapped[0]), float(mapped[1])]

            def rotated_table_quad(Htable,u,v,w,h,angle_deg):
                # First define the mock-up as a rectangle on the flat tabletop plane.
                # Rotate there, then project all four corners through the *same* homography
                # as the background tabletop. This is a genuine four-point perspective map.
                cx=u+w/2; cy=v+h/2; a=math.radians(angle_deg); ca=math.cos(a); sa=math.sin(a)
                corners=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
                out=[]
                for dx,dy in corners:
                    uu=cx+dx*ca-dy*sa; vv=cy+dx*sa+dy*ca
                    out.append(project_table_point(Htable, uu, vv))
                return out

            for mock in mockups:
                try: ensure_rgba(mock)
                except Exception as e:
                    self.log_error('load mock-up',mock,e); fail+=n; self.progress['value']+=n; continue
                for k in range(n):
                    bg=None
                    try:
                        bg=rng.choice(list(profiles.keys())); prof=profiles[bg]
                        base=Image.open(bg).convert('RGB'); W,H=base.size
                        table=prof['table']; obs=prof.get('obstacles',[])
                        # Compute one projective tabletop mapping for this background.
                        # Every mock-up is positioned in the flat plane and projected back
                        # to the original image through this same four-point transform.
                        Htable=tabletop_homography(table)
                        placed_quads=[]; warped_layers=[]
                        # Include the current mock-up, then sample the remaining mock-ups without replacement.
                        chosen=[mock]
                        chosen.extend(rng.sample([m for m in mockups if m != mock], object_count-1))
                        for mock_item in chosen:
                            fg=ensure_rgba(mock_item)
                            placed=None
                            # If a saved sample-size quad matches this mock-up's filename/class,
                            # convert that image-space quad back onto the flat tabletop plane and
                            # reuse its calibrated width/height. Otherwise use the min/max controls.
                            calibrated_wf = calibrated_hf = None
                            mock_name = mock_item.stem.lower()
                            saved_sizes = prof.get('mockup_calibrations', {})
                            match = next((key for key in sorted(saved_sizes, key=len, reverse=True)
                                          if key and key in mock_name and len(saved_sizes[key]) == 4), None)
                            if match:
                                inverse = np.linalg.inv(Htable)
                                pts = np.float32([saved_sizes[match]])
                                flat = cv2.perspectiveTransform(pts, inverse)[0]
                                topw = math.dist(flat[0], flat[1]); botw = math.dist(flat[3], flat[2])
                                lefth = math.dist(flat[0], flat[3]); righth = math.dist(flat[1], flat[2])
                                calibrated_wf = (topw + botw) / 2
                                calibrated_hf = (lefth + righth) / 2
                            for attempt in range(240):
                                # Calibrated class sizes take precedence; otherwise use width fraction controls.
                                wf = calibrated_wf if calibrated_wf is not None else rng.uniform(lo,hi)
                                if calibrated_hf is not None:
                                    hf = calibrated_hf
                                else:
                                    aspect=fg.height/max(fg.width,1)
                                    hf=wf*aspect
                                # Keep the full object inside the table's normalized coordinate space.
                                if hf>0.78 or wf>0.78: continue
                                angle=rng.uniform(-max_rotation,max_rotation) if max_rotation else 0.0
                                # Leave a margin large enough for the rotated rectangle.
                                margin=0.04+0.5*math.hypot(wf,hf)
                                if margin>=0.49: continue
                                u=rng.uniform(margin,1-margin); v=rng.uniform(margin,1-margin)
                                quad=rotated_table_quad(Htable,u-wf/2,v-hf/2,wf,hf,angle)
                                # Reject candidates whose rotated footprint extends outside the tabletop.
                                # Normalized corners are recomputed for a reliable bounds check.
                                cx=u; cy=v; a=math.radians(angle); ca=math.cos(a); sa=math.sin(a)
                                norm_corners=[(cx+dx*ca-dy*sa,cy+dx*sa+dy*ca) for dx,dy in [(-wf/2,-hf/2),(wf/2,-hf/2),(wf/2,hf/2),(-wf/2,hf/2)]]
                                if any(uu<0.02 or uu>0.98 or vv<0.02 or vv>0.98 for uu,vv in norm_corners): continue
                                if any(x<1 or y<1 or x>=W-1 or y>=H-1 for x,y in quad): continue
                                if any(polygons_overlap(quad,poly) for poly in obs if len(poly)>=3): continue
                                if any(polygons_overlap(quad,other) for other in placed_quads): continue
                                placed=quad; break
                            if placed is None:
                                raise RuntimeError(f'Could not place {mock_item.name} without overlap after 240 attempts. Mark obstacles more accurately, reduce mock-up size, or reduce mock-ups per image.')
                            placed_quads.append(placed)
                            warped=warp_rgba(fg,placed,(W,H))
                            alpha=warped.getchannel('A')
                            shadow_alpha=alpha.filter(ImageFilter.GaussianBlur(radius=max(1,int(min(W,H)*0.003))))
                            shadow_alpha=shadow_alpha.point(lambda a:int(a*0.20))
                            shadow=Image.new('RGBA',(W,H),(20,16,12,0)); shadow.putalpha(shadow_alpha)
                            shifted=Image.new('RGBA',(W,H),(0,0,0,0))
                            shifted.alpha_composite(shadow,(max(1,int(W*0.002)),max(1,int(H*0.003))))
                            warped_layers.append((shifted,warped))
                        comp=base.convert('RGBA')
                        for shadow,warped in warped_layers:
                            comp=Image.alpha_composite(comp,shadow)
                            comp=Image.alpha_composite(comp,warped)
                        comp=comp.convert('RGB')
                        name=f'{bg.stem}__{mock.stem}__{k+1:03d}_{object_count}objects.jpg'; dest=outroot/name
                        if dest.exists(): dest=outroot/f'{bg.stem}__{mock.stem}__{k+1:03d}_{object_count}objects_{rng.randrange(100000,999999)}.jpg'
                        comp.save(dest,quality=95,optimize=True); success+=1
                    except Exception as e:
                        fail+=1; errors.append(f'{mock.name} #{k+1}: {e}'); self.log_error(f'generate {mock.name} #{k+1}',bg if bg else mock,e)
                    self.progress['value']+=1; self.status.config(text=f'Generating… {int(self.progress["value"])}/{total} | success {success}, failed {fail}'); self.update_idletasks()
            self.status.config(text=f'Finished: {success} created, {fail} failed. Output: {outroot}')
            msg=f'Finished.\nCreated: {success}\nFailed: {fail}\nOutput: {outroot}'
            if errors: msg+='\n\nFirst errors:\n'+'\n'.join(errors[:5])+'\n\nSee error log for details.'
            messagebox.showinfo('Generation complete',msg)
        except Exception as e:
            self.log_error('generation setup',self.out_dir.get(),e)
            messagebox.showerror('Cannot generate',f'{e}\n\nSee error log: {self.log_path}')

if __name__=='__main__':
    try: App().mainloop()
    except Exception:
        print(traceback.format_exc())
        raise