import os
import sys
import threading
import subprocess
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Comparador de documentos Word – GUI By:Olivero"
DEFAULT_MODEL = "paraphrase-MiniLM-L6-v2"
# --- Tooltip simple para Tkinter/ttk ---
class Tooltip:
    def __init__(self, widget, text, delay=500, wraplength=420):
        self.widget = widget
        self.text = text
        self.delay = delay  # ms antes de mostrar
        self.wraplength = wraplength
        self._after_id = None
        self._tipwin = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        self._schedule()

    def _on_leave(self, _event=None):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self._after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tipwin or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if self.widget.winfo_class() == "Text" else (0, 0, 0, 0)
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2

        self._tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        frame = ttk.Frame(tw, padding=8, style="Tooltip.TFrame")
        frame.pack(fill="both", expand=True)

        label = ttk.Label(
            frame, text=self.text, justify="left",
            wraplength=self.wraplength, style="Tooltip.TLabel"
        )
        label.pack()

    def _hide(self):
        if self._tipwin:
            self._tipwin.destroy()
            self._tipwin = None

class ProcessRunner:
    def __init__(self, text_widget, on_exit=None):
        self.text_widget = text_widget
        self.on_exit = on_exit
        self.proc = None
        self.queue = queue.Queue()
        self.reader_thread = None
        self.stop_requested = False

    def append_text(self, data: str):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, data)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def _reader(self, stream, tag):
        try:
            for line in iter(stream.readline, ''):  # ya son str
                if self.stop_requested:
                    break
                self.queue.put((tag, line))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def start(self, cmd, cwd=None, env=None):
        self.stop_requested = False
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,              # <<--- fuerza modo texto
                encoding="utf-8",       # <<--- fuerza UTF-8
                errors="replace",       # reemplaza caracteres no válidos
            )
        except Exception as e:
            messagebox.showerror("Error al ejecutar", str(e))
            return

        # hilos lectores
        self.reader_thread = threading.Thread(target=self._reader, args=(self.proc.stdout, "stdout"), daemon=True)
        self.reader_thread.start()
        threading.Thread(target=self._reader, args=(self.proc.stderr, "stderr"), daemon=True).start()

        # bucle para drenar cola
        self._drain_queue()

        # monitorizar fin del proceso
        threading.Thread(target=self._wait_and_finish, daemon=True).start()

    def _wait_and_finish(self):
        rc = self.proc.wait()
        self.queue.put(("exit", rc))

    def _drain_queue(self):
        try:
            while True:
                tag, text = self.queue.get_nowait()
                if tag in ("stdout", "stderr"):
                    self.append_text(text)
                elif tag == "exit":
                    code = text
                    if self.on_exit:
                        self.on_exit(code)
                    return
        except queue.Empty:
            pass
        # reprogramar
        self.text_widget.after(50, self._drain_queue)

    def stop(self):
        self.stop_requested = True
        if self.proc and self.proc.poll() is None:
            try:
                if os.name == "nt":
                    self.proc.terminate()
                else:
                    self.proc.terminate()
            except Exception:
                pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x650")

        self.proc_runner = None
        style = ttk.Style(self)
        style.configure("Tooltip.TFrame", background="#2b2b2b")
        style.configure("Tooltip.TLabel", background="#2b2b2b", foreground="#f0f0f0")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # === Fila 1: Carpeta ===
        row = 0
        lbl_folder = ttk.Label(frm, text="Carpeta de documentos (.docx):")
        lbl_folder.grid(row=row, column=0, sticky="w")

        self.var_folder = tk.StringVar()
        ent_folder = ttk.Entry(frm, textvariable=self.var_folder, width=80)
        ent_folder.grid(row=row, column=1, sticky="we", padx=5)

        btn_pick_folder = ttk.Button(frm, text="Buscar…", command=self.pick_folder)
        btn_pick_folder.grid(row=row, column=2, sticky="w")

        self.var_recursive = tk.BooleanVar(value=False)
        chk_recursive = ttk.Checkbutton(frm, text="Incluir subcarpetas", variable=self.var_recursive)
        chk_recursive.grid(row=row, column=3, sticky="w", padx=5)

        Tooltip(lbl_folder,
                "Selecciona la carpeta que contiene los documentos Word (.docx) que quieres analizar.")
        Tooltip(ent_folder,
                "Ruta a la carpeta base del análisis. Se leerán todos los .docx que contenga.")
        Tooltip(btn_pick_folder,
                "Abrir un diálogo para elegir la carpeta con los documentos.")
        Tooltip(chk_recursive,
                "Si está activado, el comparador incluirá también todas las subcarpetas de la ruta seleccionada.")

        # === Fila 2: Salida ===
        row += 1
        lbl_out = ttk.Label(frm, text="Excel de salida:")
        lbl_out.grid(row=row, column=0, sticky="w")

        self.var_output = tk.StringVar(value="resultados_similitud.xlsx")
        ent_out = ttk.Entry(frm, textvariable=self.var_output, width=80)
        ent_out.grid(row=row, column=1, sticky="we", padx=5)

        btn_save = ttk.Button(frm, text="Guardar como…", command=self.pick_output)
        btn_save.grid(row=row, column=2, sticky="w")

        Tooltip(lbl_out, "Nombre del archivo Excel que contendrá los resultados.")
        Tooltip(ent_out,
                "Archivo .xlsx a generar. Se crearán hojas con matrices de similitud y rankings por método.\n"
                "Ejemplo: comparacion_memorias.xlsx")
        Tooltip(btn_save, "Elegir nombre y ubicación del fichero Excel de resultados.")

        # === Fila 3: Opciones (Modelo / LDA) ===
        row += 1
        lbl_model = ttk.Label(frm, text="Modelo SBERT:")
        lbl_model.grid(row=row, column=0, sticky="w")

        self.var_model = tk.StringVar(value="paraphrase-MiniLM-L6-v2")
        cbo_model = ttk.Combobox(
            frm,
            textvariable=self.var_model,
            width=40,
            values=[
                "paraphrase-MiniLM-L6-v2",
                "all-MiniLM-L12-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            state="readonly",
        )
        cbo_model.grid(row=row, column=1, sticky="w", padx=5)

        lbl_lda = ttk.Label(frm, text="Temas LDA:")
        lbl_lda.grid(row=row, column=2, sticky="e")

        self.var_lda = tk.StringVar(value="auto")
        ent_lda = ttk.Entry(frm, textvariable=self.var_lda, width=10)
        ent_lda.grid(row=row, column=3, sticky="w")

        Tooltip(lbl_model, "Modelo de embeddings para similitud semántica (detección de paráfrasis).")
        Tooltip(cbo_model,
                "Selecciona el modelo semántico para la hoja '5_Semantico_SBERT'.\n"
                "•paraphrase-MiniLM-L6-v2 → rápido y ligero. Ideal para análisis iniciales\
                o equipos con menos recursos. Precisión razonable.\n"
                "•all-mpnet-base-v2: más preciso, más lento.\n"
                "•sentece-transformers/all-MiniLM-L6-v2: equilibrado entre velocidad y precisión.\
                Adecuando para lotes de documentos medianos.\n"
                "•sentence-transformers/all-mpnet-base-v2: el más preciso, pero también el más lento."
                )
        Tooltip(lbl_lda, "Número de temas para el modelo LDA (análisis probabilístico de temas).")
        Tooltip(ent_lda,
                "Introduce un entero para fijar el nº de temas (p. ej., 10) o deja 'auto' para estimación automática.\n"
                "Afecta a la hoja '4_Probabilistico_LDA'.")

        # === Fila 4: Top K ===
        row += 1
        lbl_topk = ttk.Label(frm, text="Top K pares:")
        lbl_topk.grid(row=row, column=0, sticky="w")

        self.var_topk = tk.IntVar(value=50)
        ent_topk = ttk.Entry(frm, textvariable=self.var_topk, width=10)
        ent_topk.grid(row=row, column=1, sticky="w", padx=5)

        Tooltip(lbl_topk, "Cantidad de pares de documentos a mostrar en cada ranking 'TOP_*'.")
        Tooltip(ent_topk,
                "Número de pares más similares por método que se listarán en las hojas 'TOP_*' del Excel.")

        # === Fila 5: Script ===
        row += 1
        lbl_script = ttk.Label(frm, text="Script a ejecutar:")
        lbl_script.grid(row=row, column=0, sticky="w")

        self.var_script = tk.StringVar(value="comparadorv2.py")
        ent_script = ttk.Entry(frm, textvariable=self.var_script, width=80)
        ent_script.grid(row=row, column=1, sticky="we", padx=5)

        btn_pick_script = ttk.Button(frm, text="Buscar…", command=self.pick_script)
        btn_pick_script.grid(row=row, column=2, sticky="w")

        Tooltip(lbl_script, "Ruta del script Python que realizará el análisis.")
        Tooltip(ent_script,
                "Permite alternar entre diferentes versiones del comparador sin modificar la GUI.\n"
                "Ejemplo: comparadorv2.py")
        Tooltip(btn_pick_script, "Seleccionar el script .py a ejecutar.")

        # === Fila 6: Botones ===
        row += 1
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=4, sticky="we", pady=(10, 5))

        self.btn_run = ttk.Button(btns, text="▶ Ejecutar", command=self.run_process)
        self.btn_run.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(btns, text="⏹ Detener", command=self.stop_process, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        Tooltip(self.btn_run,
                "Inicia el análisis con las opciones configuradas.\n"
                "La salida del proceso se mostrará en la consola inferior.")
        Tooltip(self.btn_stop,
                "Envía una señal para terminar el proceso en ejecución (útil para cancelarlo).")

        # === Fila 7: Consola ===
        row += 1
        lbl_console = ttk.Label(frm, text="Consola:")
        lbl_console.grid(row=row, column=0, sticky="w", pady=(10, 0))
        Tooltip(lbl_console, "Salida en tiempo real del script (información, advertencias y errores).")

        row += 1
        self.txt = tk.Text(frm, height=20, wrap="word", state="disabled", font=("Segoe UI", 10))
        self.txt.grid(row=row, column=0, columnspan=4, sticky="nsew")

        Tooltip(self.txt,
                "Aquí verás los mensajes del script a medida que se ejecuta.\n"
                "Puedes seleccionar y copiar texto (Ctrl+C).")

        scroll = ttk.Scrollbar(frm, command=self.txt.yview)
        scroll.grid(row=row, column=4, sticky="ns")
        self.txt.configure(yscrollcommand=scroll.set)

        # === Expandir ===
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

        # === Fila 8: Progreso ===
        row += 1
        self.progress = ttk.Progressbar(frm, mode="indeterminate")
        self.progress.grid(row=row, column=0, columnspan=4, sticky="we", pady=8)
        Tooltip(self.progress, "Indicador de actividad mientras el análisis está en marcha.")


    def pick_folder(self):
        path = filedialog.askdirectory(title="Seleccionar carpeta con .docx")
        if path:
            self.var_folder.set(path)

    def pick_output(self):
        path = filedialog.asksaveasfilename(
            title="Guardar Excel de resultados",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile=self.var_output.get() or "resultados_similitud.xlsx"
        )
        if path:
            self.var_output.set(path)

    def pick_script(self):
        path = filedialog.askopenfilename(
            title="Seleccionar script comparador",
            filetypes=[("Python", "*.py"), ("Todos", "*.*")]
        )
        if path:
            self.var_script.set(path)

    def log(self, msg):
        self.txt.configure(state="normal")
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)
        self.txt.configure(state="disabled")

    def run_process(self):
        folder = self.var_folder.get().strip()
        script = self.var_script.get().strip()
        output = self.var_output.get().strip()
        model = self.var_model.get().strip()
        lda_topics = self.var_lda.get().strip()
        topk = str(self.var_topk.get())

        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Falta carpeta", "Selecciona una carpeta válida con .docx.")
            return
        if not script or not os.path.isfile(script):
            messagebox.showwarning("Falta script", "Selecciona el script a ejecutar (p.ej., comparadorv2.py).")
            return

        # construir comando
        py = sys.executable  # usa el mismo intérprete del venv
        cmd = [py, script, folder]
        if self.var_recursive.get():
            cmd.append("--recursivo")
        if output:
            cmd.extend(["--salida", output])
        if model:
            cmd.extend(["--modelo", model])
        if lda_topics and lda_topics.lower() != "auto":
            # validar entero
            try:
                int(lda_topics)
                cmd.extend(["--lda_topics", lda_topics])
            except ValueError:
                messagebox.showwarning("LDA topics", "Introduce un número entero o 'auto'.")
                return
        if topk:
            cmd.extend(["--topk", topk])

        # limpiar consola
        self.txt.configure(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.configure(state="disabled")

        self.log(f"[GUI] Ejecutando: {' '.join(f'\"{c}\"' if ' ' in c else c for c in cmd)}")
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.start(10)

        self.proc_runner = ProcessRunner(self.txt, on_exit=self.on_process_exit)
        # lanzar en hilo aparte para no bloquear
        threading.Thread(target=self.proc_runner.start, args=(cmd, None, None), daemon=True).start()

    def stop_process(self):
        if self.proc_runner:
            self.proc_runner.stop()
            self.log("[GUI] Señal de detención enviada al proceso…")

    def on_process_exit(self, return_code):
        self.progress.stop()
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if return_code == 0:
            self.log("[GUI] Proceso finalizado correctamente ✔")
        else:
            self.log(f"[GUI] Proceso finalizado con código {return_code} ⚠")

if __name__ == "__main__":
    app = App()
    app.mainloop()
