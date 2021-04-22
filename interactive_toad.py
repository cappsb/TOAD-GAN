import os
import tkinter as tk
from PIL import Image, ImageTk

# f = open("output/wandb/test4Maybe/files/arbitrary_random_samples_v1.00000_h1.00000_st0/txt/0_sc3.txt", "r")
# n = f.read()
# print(type(n))
# print(n)
# root = tk.Tk()
# myLabel = tk.Label(root, text=n)
# myLabel.pack()
# root.mainloop()

def gui_level(image1):
    test = ImageTk.PhotoImage(image1)
    label1 = tk.Label(image=test)
    label1.image = test
    # Position image
    label1.place(x=400, y=0)


