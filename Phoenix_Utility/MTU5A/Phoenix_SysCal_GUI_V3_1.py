#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phoenix SysCal V6 GUI. Python standard library only; Windows required to run SysCal."""
import datetime
import pathlib
import shutil
import subprocess
import threading
import queue
import json
import re
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

EXTS = {'.exe': 'Executable', '.pfc': 'PFC', '.tbl': 'TBL', '.clb': 'CLB', '.clc': 'CLC', '.cle': 'CLE'}

def read_text(path):
    data = pathlib.Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError('Cannot decode TBL')

class App:
    def __init__(self, root):
        self.root = root
        root.title('Phoenix SysCal V6 — Binary TBL Editor v3.1')
        root.geometry('1150x780')
        root.minsize(850, 580)
        self.folder = tk.StringVar()
        self.output = tk.StringVar()
        self.exe = tk.StringVar()
        self.pfc = tk.StringVar()
        self.tbl = tk.StringVar()
        self.cts = tk.StringVar(value='')
        self.timeout = tk.StringVar(value='120')
        self.tbl_encoding = 'cp1252'
        self.loaded_tbl = None
        self.cal_files = []
        self.parameter_rows = []
        self.binary_data = None
        self.original_binary = None
        self.changes = {}
        self.q = queue.Queue()
        panel = ttk.Frame(root, padding=10)
        panel.pack(fill='both', expand=True)
        def path_row(label, var, command):
            r = ttk.Frame(panel); r.pack(fill='x', pady=3)
            ttk.Label(r, text=label, width=17).pack(side='left')
            ttk.Entry(r, textvariable=var).pack(side='left', fill='x', expand=True)
            ttk.Button(r, text='Browse', command=command).pack(side='left', padx=4)
        path_row('Input folder', self.folder, self.pick_folder)
        path_row('Output folder', self.output, lambda: self.pick_dir(self.output))
        path_row('SysCal EXE', self.exe, lambda: self.pick_file(self.exe, [('Executable', '*.exe')]))
        path_row('PFC file', self.pfc, lambda: self.pick_file(self.pfc, [('PFC', '*.pfc')]))
        path_row('TBL file', self.tbl, self.pick_tbl)
        r = ttk.Frame(panel); r.pack(fill='x', pady=5)
        ttk.Button(r, text='Auto-detect files', command=self.scan).pack(side='left')
        ttk.Button(r, text='Load TBL', command=self.load_tbl).pack(side='left', padx=6)
        ttk.Button(r, text='Add CLB / CLC / CLE', command=self.add_cal).pack(side='left')
        ttk.Button(r, text='Remove selected', command=self.remove_cal).pack(side='left', padx=6)
        ttk.Label(r, text='CTS filename').pack(side='left', padx=(16, 4))
        ttk.Entry(r, textvariable=self.cts, width=24).pack(side='left')
        ttk.Label(r, text='Timeout (s)').pack(side='left', padx=(12, 4))
        ttk.Entry(r, textvariable=self.timeout, width=7).pack(side='left')
        # Keep the execution button visible above the long parameter table.
        r = ttk.Frame(panel); r.pack(fill='x', pady=7)
        self.run_button = ttk.Button(r, text='Run SysCal → CTS', command=self.run)
        self.run_button.pack(side='left')
        ttk.Button(r, text='Save edited TBL as...', command=self.save_tbl).pack(side='left', padx=8)
        ttk.Label(r, text='Original source files are never overwritten.').pack(side='left', padx=12)
        ttk.Label(panel, text='Calibration files (CLB / CLC / CLE):').pack(anchor='w')
        self.cal_list = tk.Listbox(panel, height=5, selectmode='extended')
        self.cal_list.pack(fill='x', pady=(2, 8))
        ttk.Label(panel, text='Binary TBL fields — double-click Value to edit (changes applied to a binary copy):').pack(anchor='w')
        params_frame = ttk.Frame(panel); params_frame.pack(fill='x', pady=(2, 6))
        self.params = ttk.Treeview(params_frame, columns=('line', 'key', 'value'), show='headings', height=13)
        for col, width in [('line', 55), ('key', 360), ('value', 300)]:
            self.params.heading(col, text=col.capitalize()); self.params.column(col, width=width, stretch=(col != 'line'))
        self.params.pack(side='left', fill='x', expand=True)
        pscroll = ttk.Scrollbar(params_frame, orient='vertical', command=self.params.yview)
        pscroll.pack(side='left', fill='y'); self.params.configure(yscrollcommand=pscroll.set)
        self.params.bind('<Double-1>', self.edit_parameter)
        actions = ttk.Frame(panel); actions.pack(fill='x', pady=(0, 5))
        ttk.Button(actions, text='Refresh parameters from TBL', command=self.refresh_parameters).pack(side='left')
        ttk.Button(actions, text='Find gain / spacing', command=self.find_parameter).pack(side='left', padx=7)
        ttk.Label(actions, text='Gain codes are not necessarily physical gain multipliers.').pack(side='left', padx=8)
        ttk.Label(panel, text='Binary record inspection (read-only; edits must use the table above):').pack(anchor='w')
        frame = ttk.Frame(panel); frame.pack(fill='both', expand=True)
        self.editor = tk.Text(frame, wrap='none', undo=False, height=12, font=('Consolas', 10), state='disabled')
        sy = ttk.Scrollbar(frame, orient='vertical', command=self.editor.yview)
        sx = ttk.Scrollbar(frame, orient='horizontal', command=self.editor.xview)
        self.editor.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.editor.grid(row=0, column=0, sticky='nsew'); sy.grid(row=0, column=1, sticky='ns'); sx.grid(row=1, column=0, sticky='ew')
        frame.grid_columnconfigure(0, weight=1); frame.grid_rowconfigure(0, weight=1)
        ttk.Label(panel, text='Execution log:').pack(anchor='w')
        self.logbox = tk.Text(panel, height=9, wrap='word', font=('Consolas', 9))
        self.logbox.pack(fill='both')
        self.root.after(100, self.poll)
    def log(self, message):
        self.logbox.insert('end', str(message) + '\n'); self.logbox.see('end')
    def pick_dir(self, var):
        p = filedialog.askdirectory(initialdir=var.get() or None)
        if p: var.set(p)
    def pick_folder(self):
        self.pick_dir(self.folder)
        if self.folder.get(): self.scan()
    def pick_file(self, var, types):
        p = filedialog.askopenfilename(initialdir=self.folder.get() or None, filetypes=types + [('All files', '*.*')])
        if p: var.set(p)
    def pick_tbl(self):
        self.pick_file(self.tbl, [('TBL', '*.tbl')]); self.load_tbl()
    def set_cal(self, paths):
        self.cal_files = list(dict.fromkeys(str(pathlib.Path(p).resolve()) for p in paths))
        self.cal_list.delete(0, 'end')
        for p in self.cal_files: self.cal_list.insert('end', p)
    def add_cal(self):
        paths = filedialog.askopenfilenames(initialdir=self.folder.get() or None, filetypes=[('Calibration', '*.clb *.clc *.cle'), ('All', '*.*')])
        self.set_cal(self.cal_files + list(paths))
    def remove_cal(self):
        removed = set(self.cal_list.curselection())
        self.set_cal([p for i, p in enumerate(self.cal_files) if i not in removed])
    def scan(self):
        base = pathlib.Path(self.folder.get())
        if not base.is_dir(): messagebox.showerror('Input', 'Choose a valid input folder.'); return
        paths = [p for p in base.iterdir() if p.is_file()]
        def one(ext, preferred=None):
            found = sorted((p for p in paths if p.suffix.lower() == ext), key=lambda p: p.name.lower())
            if preferred:
                priority = [p for p in found if preferred in p.name.lower()]
                if priority: return str(priority[0])
            return str(found[0]) if len(found) == 1 else ''
        self.exe.set(one('.exe', 'syscal'))
        self.pfc.set(one('.pfc', 'default'))
        tbls = sorted((p for p in paths if p.suffix.lower() == '.tbl'), key=lambda p:p.name.lower())
        self.tbl.set(str(tbls[0]) if tbls else '')
        cal = [p for p in paths if p.suffix.lower() in ('.clb', '.clc', '.cle')]
        for sub in ('receiver', 'coils'):
            d = base / sub
            if d.is_dir(): cal.extend(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ('.clb', '.clc', '.cle'))
        self.set_cal(cal)
        if not self.output.get(): self.output.set(str(base.parent / (base.name + '_SysCal_Output')))
        self.load_tbl()
        self.log(f'Scanned {base}: {len(cal)} calibration files; select ambiguous EXE/PFC/TBL manually.')
    def load_tbl(self):
        p = pathlib.Path(self.tbl.get())
        if not p.is_file(): return
        try:
            data = p.read_bytes()
            if len(data) % 25 != 0 or len(data) < 250:
                raise ValueError('Unsupported TBL layout: expected 25-byte binary records.')
            self.binary_data = bytearray(data)
            self.original_binary = data
            self.loaded_tbl = str(p.resolve())
            self.changes = {}
            self.cts.set(p.stem.split('(')[0] + '.CTS')
            self.refresh_parameters()
            self.log(f'Loaded binary TBL: {p.name}, {len(data)//25} records, {len(data)} bytes.')
        except Exception as e: messagebox.showerror('TBL read error', str(e))

    def records(self):
        if self.binary_data is None: return []
        rows = []
        for i in range(len(self.binary_data)//25):
            offset = i*25
            raw = self.binary_data[offset:offset+25]
            key = bytes(raw[:4]).rstrip(b'\x00').decode('ascii','replace')
            kind = raw[11]
            if kind == 1:
                value = format(struct.unpack_from('<d',raw,12)[0],'.12g')
                dtype = 'float64'
            elif kind in (2,4):
                value = bytes(raw[12:25]).split(b'\x00')[0].decode('ascii','replace')
                dtype = 'text'
            elif kind == 5:
                value = bytes(raw[12:25]).hex(' ')
                dtype = 'bytes'
            else:
                value = str(struct.unpack_from('<i',raw,12)[0]);dtype='int32'
            rows.append((i,key,kind,dtype,value))
        return rows

    def refresh_parameters(self):
        self.params.delete(*self.params.get_children())
        self.parameter_rows = self.records()
        for i,key,kind,dtype,value in self.parameter_rows:
            label = key + ('  [electrode length, m]' if key in ('EXLN','EYLN') else '  [gain code / setting]' if key in ('EGNC','HGNC','EGN','HGN') else '')
            self.params.insert('', 'end', iid=str(i), values=(i, label, value))
        lines = [f'{i:03d}  {key:4s}  {dtype:7s}  {value}' for i,key,kind,dtype,value in self.parameter_rows]
        self.editor.configure(state='normal')
        self.editor.delete('1.0','end');self.editor.insert('1.0','\n'.join(lines))
        self.editor.configure(state='disabled')

    def edit_parameter(self, event):
        item = self.params.identify_row(event.y)
        if not item or self.params.identify_column(event.x) != '#3': return
        from tkinter import simpledialog
        i = int(item)
        _,key,kind,dtype,old = self.parameter_rows[i]
        if dtype == 'bytes':
            messagebox.showinfo('Read only','Complex binary timestamp/byte field is read-only.');return
        new = simpledialog.askstring('Edit binary TBL',f'{key} ({dtype})\nOriginal/current: {old}\nNew value:',initialvalue=old,parent=self.root)
        if new is None or new.strip() == old: return
        new = new.strip()
        try:
            offset = i*25+12
            if dtype == 'float64':
                import math
                value = float(new)
                if not math.isfinite(value): raise ValueError('Value must be finite')
                if key in ('EXLN','EYLN') and value <= 0: raise ValueError('Electrode length must be positive')
                struct.pack_into('<d',self.binary_data,offset,value)
            elif dtype == 'int32':
                value = int(new)
                struct.pack_into('<i',self.binary_data,offset,value)
            else:
                encoded = new.encode('ascii')
                if len(encoded)>13: raise ValueError('Text exceeds 13 bytes')
                self.binary_data[offset:offset+13] = encoded.ljust(13,b'\x00')
            self.changes[key] = {'record':i,'from':self.original_value(i),'to':new}
            self.refresh_parameters()
            self.log(f'Edited {key}: {old} -> {new} (record {i})')
        except Exception as e: messagebox.showerror('Invalid edit',str(e))

    def original_value(self,i):
        raw=self.original_binary[i*25:(i+1)*25]
        kind=raw[11]
        if kind==1:return struct.unpack_from('<d',raw,12)[0]
        if kind in (2,4):return raw[12:25].split(b'\x00')[0].decode('ascii','replace')
        return struct.unpack_from('<i',raw,12)[0]

    def find_parameter(self):
        from tkinter import simpledialog
        term=simpledialog.askstring('Find binary field','Field name (e.g. EXLN, EYLN, EGN, HGN):',initialvalue='EXLN',parent=self.root)
        if not term:return
        for i,key,*_ in self.parameter_rows:
            if term.lower() in key.lower():
                self.params.selection_set(str(i));self.params.see(str(i));self.params.focus(str(i));return
        messagebox.showinfo('Not found',f'No matching TBL field: {term}')

    def save_tbl(self):
        if self.binary_data is None:return
        p=filedialog.asksaveasfilename(defaultextension='.TBL',filetypes=[('TBL','*.TBL')])
        if p:
            if pathlib.Path(p).resolve()==pathlib.Path(self.tbl.get()).resolve():
                messagebox.showerror('Source protection','Cannot overwrite the original TBL.');return
            pathlib.Path(p).write_bytes(self.binary_data)
            self.log('Saved binary TBL: '+p)

    def run(self):
        try:
            if __import__('sys').platform != 'win32': raise ValueError('SysCal-V6.exe can only run on Windows.')
            exe, pfc, tbl = (pathlib.Path(v.get()).resolve() for v in (self.exe, self.pfc, self.tbl))
            for p in (exe, pfc, tbl):
                if not p.is_file(): raise ValueError(f'Missing file: {p}')
            if self.loaded_tbl != str(tbl): raise ValueError('Click Load TBL before running, to ensure the edited text matches the selected file.')
            if not self.cal_files: raise ValueError('Select CLB/CLC calibration files.')
            cals = [pathlib.Path(p) for p in self.cal_files]
            for p in cals:
                if not p.is_file(): raise ValueError(f'Missing calibration file: {p}')
            if not any(p.suffix.lower() == '.clb' for p in cals): raise ValueError('At least one receiver CLB is required.')
            if not any(p.suffix.lower() == '.clc' for p in cals): raise ValueError('At least one coil CLC is required.')
            output = pathlib.Path(self.output.get()).resolve()
            if not self.output.get(): raise ValueError('Select an output folder.')
            if output == pathlib.Path(self.folder.get()).resolve(): raise ValueError('Output folder must differ from input folder.')
            name = self.cts.get().strip()
            if not name or pathlib.Path(name).name != name or not name.lower().endswith('.cts'): raise ValueError('CTS filename must be a simple name ending in .CTS.')
            timeout = int(self.timeout.get()); assert timeout > 0
            edited = bytes(self.binary_data) if self.binary_data is not None else None
            if edited is None: raise ValueError('Load binary TBL first.')
            output.mkdir(parents=True, exist_ok=True)
            work = output / ('syscal_work_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            work.mkdir(); (work / 'receiver').mkdir(); (work / 'coils').mkdir()
            shutil.copy2(exe, work / exe.name); shutil.copy2(pfc, work / pfc.name)
            (work / tbl.name).write_bytes(edited)
            used = set()
            for p in cals:
                sub = 'receiver' if p.suffix.lower() == '.clb' else 'coils'
                dest = work / sub / p.name
                if str(dest).lower() in used: raise ValueError(f'Duplicate calibration filename: {p.name}')
                used.add(str(dest).lower()); shutil.copy2(p, dest)
            cmd = [str(work / exe.name), pfc.name, tbl.name, r'receiver\*.CLB', r'coils\*.CLC', name]
            manifest = {'source_tbl': str(tbl), 'source_exe': str(exe), 'source_pfc': str(pfc), 'calibration_files': [str(p) for p in cals], 'output_cts': name, 'tbl_changes': self.changes, 'command': cmd, 'note': 'CLE copied to coils directory but original SysCal V6 invocation passes only CLC wildcard; CLE consumption is unverified.'}
            (work / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            self.run_button.configure(state='disabled')
            self.log('Running in: ' + str(work))
            threading.Thread(target=self.worker, args=(cmd, work, output, name, timeout), daemon=True).start()
        except Exception as e: messagebox.showerror('Validation error', str(e))
    def worker(self, cmd, work, output, name, timeout):
        try:
            proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, errors='replace', timeout=timeout, shell=False)
            log = f'UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\nCommand: {subprocess.list2cmdline(cmd)}\nReturn code: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n'
            (work / 'syscal_run.log').write_text(log, encoding='utf-8')
            generated = work / name
            if not generated.is_file():
                matches = list(work.glob('*.CTS'))
                if len(matches) == 1: generated = matches[0]
            if proc.returncode == 0 and generated.is_file() and generated.stat().st_size > 0:
                dest = output / name
                shutil.copy2(generated, dest)
                self.q.put(('success', f'CTS created: {dest}\nWorking files: {work}\n' + log))
            else: self.q.put(('error', 'SysCal did not produce a valid CTS.\n' + log + '\nWorking files: ' + str(work)))
        except Exception as e: self.q.put(('error', f'Execution error: {e}\nWorking files: {work}'))
    def poll(self):
        try:
            while True:
                status, msg = self.q.get_nowait(); self.log(msg); self.run_button.configure(state='normal')
                if status == 'error': messagebox.showerror('SysCal error', msg[:1500])
                else: messagebox.showinfo('Completed', msg[:500])
        except queue.Empty: pass
        self.root.after(100, self.poll)

if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
