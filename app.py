import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading

from utils.detector import detect_and_annotate
from utils.model_loader import load_model


class CarDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Colour Detection")
        self.root.configure(bg="#0f0f13")
        self.root.geometry("1100x650")

        self.img = None
        self.result = None
        self.model = None

        self.setup_ui()
        self.load_model_bg()

    def setup_ui(self):
        top = tk.Frame(self.root, bg="#0f0f13")
        top.pack(fill="x", padx=20, pady=(16, 4))

        tk.Label(top, text="Car Colour Detection",
                 font=("Courier New", 18, "bold"),
                 bg="#0f0f13", fg="#e2e8f0").pack(side="left")

        tk.Label(top, text="YOLOv8",
                 font=("Courier New", 9),
                 bg="#0f0f13", fg="#4a5568").pack(side="left", padx=10)

        self.model_status = tk.Label(top, text="loading model...",
                                     font=("Courier New", 9),
                                     bg="#2d2d3d", fg="#f6ad55",
                                     padx=8, pady=3)
        self.model_status.pack(side="right")

        btn_row = tk.Frame(self.root, bg="#0f0f13")
        btn_row.pack(pady=8)

        tk.Button(btn_row, text="Load Image",  command=self.load_image,
                  bg="#63b3ed", fg="#0f0f13", font=("Courier New", 9, "bold"),
                  relief="flat", padx=14, pady=7, cursor="hand2").pack(side="left", padx=6)

        tk.Button(btn_row, text="Analyse",     command=self.run_detection,
                  bg="#68d391", fg="#0f0f13", font=("Courier New", 9, "bold"),
                  relief="flat", padx=14, pady=7, cursor="hand2").pack(side="left", padx=6)

        tk.Button(btn_row, text="Save Result", command=self.save,
                  bg="#f6ad55", fg="#0f0f13", font=("Courier New", 9, "bold"),
                  relief="flat", padx=14, pady=7, cursor="hand2").pack(side="left", padx=6)

        tk.Button(btn_row, text="Clear",       command=self.clear,
                  bg="#718096", fg="#0f0f13", font=("Courier New", 9, "bold"),
                  relief="flat", padx=14, pady=7, cursor="hand2").pack(side="left", padx=6)

        panels = tk.Frame(self.root, bg="#0f0f13")
        panels.pack(fill="both", expand=True, padx=16, pady=4)

        left = tk.Frame(panels, bg="#1a1a2e")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(left, text="Input", font=("Courier New", 9),
                 bg="#1a1a2e", fg="#4a5568").pack(pady=(8, 4))
        self.lbl_in = tk.Label(left, bg="#1a1a2e", text="no image loaded",
                               fg="#4a5568", font=("Courier New", 10))
        self.lbl_in.pack(expand=True)

        right = tk.Frame(panels, bg="#1a1a2e")
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))
        tk.Label(right, text="Result", font=("Courier New", 9),
                 bg="#1a1a2e", fg="#4a5568").pack(pady=(8, 4))
        self.lbl_out = tk.Label(right, bg="#1a1a2e", text="run analyse to see result",
                                fg="#4a5568", font=("Courier New", 10))
        self.lbl_out.pack(expand=True)

        stats_bg = tk.Frame(self.root, bg="#1a1a2e")
        stats_bg.pack(fill="x", padx=16, pady=(4, 0))

        stat_row = tk.Frame(stats_bg, bg="#1a1a2e")
        stat_row.pack(fill="x", padx=12, pady=8)

        self.v_total  = tk.StringVar(value="0")
        self.v_blue   = tk.StringVar(value="0")
        self.v_other  = tk.StringVar(value="0")
        self.v_people = tk.StringVar(value="0")

        self.make_stat_box(stat_row, "Total Cars",  self.v_total,  "#63b3ed", 0)
        self.make_stat_box(stat_row, "Blue Cars",   self.v_blue,   "#f56565", 1)
        self.make_stat_box(stat_row, "Other Cars",  self.v_other,  "#63b3ed", 2)
        self.make_stat_box(stat_row, "People",      self.v_people, "#68d391", 3)

        self.colour_text = tk.Label(stats_bg, text="",
                                    font=("Courier New", 8),
                                    bg="#1a1a2e", fg="#718096")
        self.colour_text.pack(pady=(0, 6))

        self.status = tk.StringVar(value="load an image to get started")
        tk.Label(self.root, textvariable=self.status,
                 font=("Courier New", 8), bg="#0a0a0f", fg="#718096",
                 anchor="w", padx=12).pack(fill="x", side="bottom", ipady=4)

    def make_stat_box(self, parent, label, var, color, col):
        box = tk.Frame(parent, bg="#0f0f13", padx=16, pady=8)
        box.grid(row=0, column=col, padx=6, pady=4, sticky="ew")
        parent.columnconfigure(col, weight=1)
        tk.Label(box, textvariable=var,
                 font=("Courier New", 22, "bold"),
                 bg="#0f0f13", fg=color).pack()
        tk.Label(box, text=label,
                 font=("Courier New", 8),
                 bg="#0f0f13", fg="#4a5568").pack()

    def load_model_bg(self):
        # run in thread so the window doesnt freeze while downloading
        def _do():
            self.status.set("downloading yolo weights, please wait...")
            self.model = load_model()
            if self.model:
                self.model_status.config(text="model ready", fg="#68d391")
                self.status.set("model ready — load an image and click analyse")
            else:
                self.model_status.config(text="model failed", fg="#f56565")
                self.status.set("model failed to load. check internet connection")

        threading.Thread(target=_do, daemon=True).start()

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        self.img = cv2.imread(path)
        if self.img is None:
            messagebox.showerror("Error", "could not open that image")
            return

        self.result = None
        self.show_img(self.lbl_in, self.img)
        self.lbl_out.config(image="", text="run analyse to see result")
        self.reset_stats()
        self.status.set(f"loaded: {os.path.basename(path)}")

    def run_detection(self):
        if self.img is None:
            messagebox.showinfo("No image", "load an image first")
            return
        if self.model is None:
            messagebox.showinfo("Not ready", "model is still loading, wait a moment")
            return

        self.status.set("running detection...")
        self.root.update()

        self.result, stats = detect_and_annotate(self.model, self.img)
        self.show_img(self.lbl_out, self.result)

        self.v_total.set(str(stats["total_cars"]))
        self.v_blue.set(str(stats["blue_cars"]))
        self.v_other.set(str(stats["other_cars"]))
        self.v_people.set(str(stats["people"]))

        cc = stats.get("color_counts", {})
        if cc:
            self.colour_text.config(
                text="  |  ".join(f"{k}: {v}" for k, v in sorted(cc.items()))
            )

        self.status.set(
            f"done — {stats['total_cars']} cars, "
            f"{stats['blue_cars']} blue (red box), "
            f"{stats['other_cars']} other (blue box), "
            f"{stats['people']} people"
        )

    def save(self):
        if self.result is None:
            messagebox.showinfo("Nothing to save", "run analyse first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if path:
            cv2.imwrite(path, self.result)
            self.status.set(f"saved to {path}")

    def clear(self):
        self.img = None
        self.result = None
        self.lbl_in.config(image="", text="no image loaded")
        self.lbl_out.config(image="", text="run analyse to see result")
        self.reset_stats()
        self.status.set("cleared")

    def show_img(self, label, img, max_w=500, max_h=360):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ph = ImageTk.PhotoImage(Image.fromarray(rgb))
        label._ph = ph  # needed or tkinter garbage collects the image
        label.config(image=ph, text="")

    def reset_stats(self):
        self.v_total.set("0")
        self.v_blue.set("0")
        self.v_other.set("0")
        self.v_people.set("0")
        self.colour_text.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    CarDetectorApp(root)
    root.mainloop()
