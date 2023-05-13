import subprocess
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from sklearn.linear_model import LinearRegression

image = None
image_resized = None
image_id = None
image_zoom = None

start_x, start_y = 0, 0
rect_id = None
rect_coords = (0, 0, 0, 0)
rgb_mean = []
s_select = []
concentration = [0, 10, 20, 40, 60, 80, 100]
pixeltrain = []
pixeltest = []
x_pred = None

model = LinearRegression()

# 图片缩放比例
zoom_scale = 1.0

# 图片缩放程度
zoom_step = 0.2


# 打开图片，并在 GUI 界面显示
def select_image():
    global image, image_resized, image_id, image_zoom
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    # resized
    image_resized = image.resize((600, 450))
    image_zoom = image_resized
    # Image to PhotoImage
    photo = ImageTk.PhotoImage(image_resized)
    # show
    photo_canvas.itemconfig(image_id, image=photo)
    photo_canvas.image = photo
    # image_id = photo_canvas.create_image(200, 150, anchor='center', image=photo)
    # photo_canvas.config(scrollregion=photo_canvas.bbox(tk.ALL))
    photo_canvas.bind("<ButtonPress-3>", on_start_drag)
    photo_canvas.bind("<B3-Motion>", on_drag)

    # photo_label.config(image=photo)
    # photo_label.image = photo
    # button
    btn_zoom_in.config(state=tk.NORMAL)
    btn_reset.config(state=tk.NORMAL)
    btn_zoom_out.config(state=tk.NORMAL)
    btn_start.config(state=tk.NORMAL)
    btn_processing.config(state=tk.NORMAL)
    btn_select.config(state=tk.NORMAL)


def zoom_out():
    global zoom_scale, image_zoom, image_resized
    zoom_scale -= zoom_step
    if zoom_scale < 0.4:
        zoom_scale = 0.4
    image_zoom = image_zoom.resize((int(image_resized.width * zoom_scale), int(image_resized.height * zoom_scale)))
    photo_zoom_out = ImageTk.PhotoImage(image_zoom)
    photo_canvas.itemconfig(image_id, image=photo_zoom_out)
    photo_canvas.image = photo_zoom_out
    photo_canvas.bind("<ButtonPress-3>", on_start_drag)
    photo_canvas.bind("<B3-Motion>", on_drag)


def zoom_in():
    global zoom_scale, image_zoom, image_resized
    zoom_scale += zoom_step
    if zoom_scale > 5.0:
        zoom_scale = 5.0
    image_zoom = image_zoom.resize((int(image_resized.width * zoom_scale), int(image_resized.height * zoom_scale)))
    photo_zoom_in = ImageTk.PhotoImage(image_zoom)
    photo_canvas.itemconfig(image_id, image=photo_zoom_in)
    photo_canvas.image = photo_zoom_in
    photo_canvas.bind("<ButtonPress-3>", on_start_drag)
    photo_canvas.bind("<B3-Motion>", on_drag)


def reset():
    global zoom_scale, image_id, image_zoom, image_resized
    zoom_scale = 1.0
    image_zoom = image_zoom.resize((int(image_resized.width * zoom_scale), int(image_resized.height * zoom_scale)))
    photo_reset = ImageTk.PhotoImage(image_zoom)

    # photo_canvas.delete(image_id)
    # image_id = photo_canvas.create_image(200, 150, anchor='center', image=photo_reset)

    photo_canvas.itemconfig(image_id, image=photo_reset, anchor='center')
    photo_canvas.image = photo_reset
    photo_canvas.bind("<ButtonPress-3>", on_start_drag)
    photo_canvas.bind("<B3-Motion>", on_drag)


def progress_bar():
    for i in range(100):
        progress['value'] = i
        progress_label.config(text=f'{i}%')
        progress_frame.update()
        time.sleep(0.15)


def processing():
    global image_zoom
    image_cv = cv2.cvtColor(np.array(image_zoom), cv2.COLOR_RGB2BGR)
    image_mean = cv2.blur(image_cv, (5, 5))
    image_gauss = cv2.GaussianBlur(image_mean, (3, 3), 1)
    image_gaussian = Image.fromarray(cv2.cvtColor(image_gauss, cv2.COLOR_BGR2RGB))
    photo_filter = ImageTk.PhotoImage(image_gaussian)
    photo_canvas.itemconfig(image_id, image=photo_filter)
    photo_canvas.image = photo_filter


def on_mouse_down(event):
    global start_x, start_y
    start_x, start_y = event.x, event.y


def on_mouse_move(event):
    global rect_id, rect_coords
    if rect_id:
        photo_canvas.delete(rect_id)
    x1, y1 = start_x, start_y
    x2, y2 = event.x, event.y
    rect_coords = (x1, y1, x2, y2)
    rect_id = photo_canvas.create_rectangle(x1, y1, x2, y2, outline="red")


def on_mouse_up(event):
    global rect_id, rgb_mean, image_zoom, rgb_mean
    x1, y1, x2, y2 = rect_coords
    if x1 < x2 and y1 < y2:
        # RGB
        rgb_values = []
        for x in range(x1, x2):
            for y in range(y1, y2):
                color = image_zoom.getpixel((x, y))
                rgb_values.append(color)
        # mean
        r_sum = sum([r for r, g, b in rgb_values]) // len(rgb_values)
        g_sum = sum([g for r, g, b in rgb_values]) // len(rgb_values)
        b_sum = sum([b for r, g, b in rgb_values]) // len(rgb_values)
        rgb_mean.append((r_sum, g_sum, b_sum))
        messagebox.showinfo("Tips", "The average value of RGB is " + str(rgb_mean[-1]))
        print(rgb_mean)
        r_trans = r_sum / 255
        g_trans = g_sum / 255
        b_trans = b_sum / 255
        c_max = max(r_trans, g_trans, b_trans)
        c_min = min(r_trans, g_trans, b_trans)
        delta = c_max - c_min
        if c_max == 0:
            s = 0
        else:
            s = delta / c_max
        s_select.append(s)
        print(s_select)
    else:
        messagebox.showerror("Error", "Please select a rectangular box larger than 0 * 0")
    rect_id = None


def color_selection():
    photo_canvas.bind("<Button-1>", on_mouse_down)
    photo_canvas.bind("<B1-Motion>", on_mouse_move)
    photo_canvas.bind("<ButtonRelease-1>", on_mouse_up)


def start():
    global concentration, x_pred, s_select, model, pixeltrain, pixeltest
    # button
    btn_start.config(state=tk.DISABLED)
    btn_zoom_out.config(state=tk.DISABLED)
    btn_reset.config(state=tk.DISABLED)
    btn_zoom_in.config(state=tk.DISABLED)
    btn_processing.config(state=tk.DISABLED)
    btn_select.config(state=tk.DISABLED)
    btn_show.config(state=tk.DISABLED)
    # progress_bar
    subprocess.Popen("command1", shell=True)
    progress_bar()
    subprocess.Popen("command2", shell=True)
    concentration = np.array(concentration).reshape((-1, 1))

    pixeltrain = [s_select[0], s_select[1], s_select[2], s_select[3], s_select[4], s_select[5], s_select[6]]
    pixeltest = s_select[-1]
    print(pixeltrain)
    model.fit(concentration, pixeltrain)
    r_sq = model.score(concentration, pixeltrain)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    x_pred = (pixeltest - model.intercept_) / model.coef_
    print('predicted concentration:', x_pred)

    progress['value'] = 100
    progress_label.config(text='100%')
    progress_frame.update()
    time.sleep(0.15)
    progress_label.config(text='Finish !')
    result_label.config(text='So, the concentration of your glucose solution is', font=('Arial', 12, 'italic'),
                        height=2)
    result.config(text=x_pred, fg='blue', font=('Arial', 15, 'bold italic'), height=2)
    result_label2.config(text='mM !', font=('Arial', 12, 'italic'), height=2)

    # button
    btn_start.config(state=tk.NORMAL)
    btn_show.config(state=tk.NORMAL)


def on_start_drag(event):
    photo_canvas.scan_mark(event.x, event.y)


def on_drag(event):
    photo_canvas.scan_dragto(event.x, event.y, gain=1)


def show():
    global model, concentration, pixeltrain, pixeltest
    width = np.linspace(0, 100, 1000)
    height = model.coef_ * width + model.intercept_
    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(concentration, pixeltrain)
    plt.plot(width, height)
    plt.scatter([x_pred], [pixeltest], s=25, c='r')
    plt.show()


# GUI
window = tk.Tk()
window.title('Colorimetric Biosensor for Glucose Detection')
window.geometry('750x750')
window.resizable(False, False)

# title
title = tk.Label(window, text='Colorimetric Biosensor for Glucose Detection', font=('Times', 18, 'bold italic'),
                 height=2)
title.pack()

# button
btn = tk.Button(window, text="Open the photo of your glucose solution", command=select_image)
btn.pack()

# photo
photo_canvas = tk.Canvas(window, width=600, height=450)
photo_canvas.pack(pady=5)
photo_canvas.config(scrollregion=photo_canvas.bbox(tk.ALL))
# photo_canvas.bind("<ButtonPress-1>", on_start_drag)
# photo_canvas.bind("<B1-Motion>", on_drag)


image_default = Image.open('mctu_logo.png')
image_default = image_default.resize((200, 200))
photo_default = ImageTk.PhotoImage(image_default)
image_id = photo_canvas.create_image(300, 225, anchor='center', image=photo_default)

# photo_label = tk.Label(window, image=photo_default, height=300, width=400)
# photo_label.pack(pady=10)

# Frame
btn_frame = tk.Frame(window, width=700)
btn_frame.pack()

# zoom_in
btn_zoom_in = tk.Button(btn_frame, text='zoom in', width=10, command=zoom_in, state=tk.DISABLED)
btn_zoom_in.pack(side='left', padx=30)

# reset
btn_reset = tk.Button(btn_frame, text='reset', width=10, command=reset, state=tk.DISABLED)
btn_reset.pack(side='left', padx=30)

# zoom_out
btn_zoom_out = tk.Button(btn_frame, text='zoom out', width=10, command=zoom_out, state=tk.DISABLED)
btn_zoom_out.pack(side='right', padx=30)

# Frame
func_frame = tk.Frame(window, width=700)
func_frame.pack(pady=15)

# processing
btn_processing = tk.Button(func_frame, text='processing', width=10, command=lambda: processing(), state=tk.DISABLED)
btn_processing.pack(side='left', padx=20)

# color selection
btn_select = tk.Button(func_frame, text='select', width=10, command=lambda: color_selection(), state=tk.DISABLED)
btn_select.pack(side='left', padx=20)

# start
btn_start = tk.Button(func_frame, text='start', width=10, command=lambda: start(), state=tk.DISABLED)
btn_start.pack(side='left', padx=20)

# show
btn_show = tk.Button(func_frame, text='show', width=10, command=lambda: show(), state=tk.DISABLED)
btn_show.pack(side='right', padx=20)

# Frame
progress_frame = tk.Frame(window, width=750)
progress_frame.pack(pady=5)

# progress
progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=250, mode='determinate')
progress.pack(side='left', padx=20)

progress_label = tk.Label(progress_frame, text='0%', width=8)
progress_label.pack(side='right', padx=20)

# Frame
result_frame = tk.Frame(window, width=700)
result_frame.pack(pady=5)

# result and label
result_label = tk.Label(result_frame, text='', height=2)
result_label.pack(side='left', ipadx=50, ipady=100)
result = tk.Label(result_frame, text='', height=2)
result.pack(side='left')
result_label2 = tk.Label(result_frame, text='', height=2)
result_label2.pack(side='left', ipadx=50)

window.mainloop()
