# 创建GUI窗口打开图像 并显示在窗口中

from PIL import Image, ImageTk # 导入图像处理函数库
import tkinter as tk           # 导入GUI界面函数库
import os
import shutil
import tkinter.messagebox #这个是消息框，对话框的关键
i=1
# 创建窗口 设定大小并命名
window = tk.Tk()
window.title(i)
window.geometry('1000x1000')
img_png=None           # 定义全局变量 图像的
var = tk.StringVar()    # 这时文字变量储存器
simple_path=r"G:\select\simple"
complex_path=r"G:\select\complex"
middle_path=r"G:\select\middle"
path=r'G:\select\complex'
files=os.listdir(path)
photo=None
img=None
Img = Image.open(os.path.join(path, files[0]))
Img = Img.resize((700, 700), resample=0)
img = ImageTk.PhotoImage(Img)
img_png = tk.Label(window, image=img)
img_png.pack()


def simple():
    global img_png
    global i
    i += 1
    print(files[i])
    Img = Image.open(os.path.join(path, files[i]))
    Img = Img.resize((700, 700), resample=0)
    img = ImageTk.PhotoImage(Img)
    img_png.configure(image=img)  # 这里的img是tk图片对象，button是创建好的按钮对象
    img_png.image = img
    shutil.move(os.path.join(path,files[i]),os.path.join(simple_path,files[i]))
def middle():
    global img_png
    global i
    i += 1
    print(files[i])
    Img = Image.open(os.path.join(path, files[i]))
    Img = Img.resize((700, 700), resample=0)
    img = ImageTk.PhotoImage(Img)
    img_png.configure(image=img)  # 这里的img是tk图片对象，button是创建好的按钮对象
    img_png.image = img
    shutil.move(os.path.join(path,files[i]),os.path.join(middle_path,files[i]))
def complex():
    global img_png
    global i
    i += 1
    print(files[i])
    Img = Image.open(os.path.join(path, files[i]))
    Img = Img.resize((700, 700), resample=0)
    img = ImageTk.PhotoImage(Img)
    img_png.configure(image=img)  # 这里的img是tk图片对象，button是创建好的按钮对象
    img_png.image = img
    shutil.move(os.path.join(path,files[i]),os.path.join(complex_path,files[i]))

# 创建打开图像按钮
btn_simple = tk.Button(window,
    text='简单',      # 显示在按钮上的文字
    width=15, height=2,
    command=simple)     # 点击按钮式执行的命令
btn_simple.pack()    # 按钮位置
btn_middle = tk.Button(window,
    text='中等',      # 显示在按钮上的文字
    width=15, height=2,
    command=middle)     # 点击按钮式执行的命令
btn_middle.pack()    # 按钮位置
btn_complex = tk.Button(window,
    text='复杂',      # 显示在按钮上的文字
    width=15, height=2,
    command=complex)     # 点击按钮式执行的命令
btn_complex.pack()    # 按钮位置
# 运行整体窗口
window.mainloop()
