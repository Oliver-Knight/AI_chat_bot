import tkinter as tk
from tkinter import *
import os
from tkinter import messagebox
from run_bot_function import *
BG = '#3d3d3d'
TEXT = '#ebebeb'
root = Tk()
root.title("Py Chat_bot")
frame = Frame(root, width=600, height=700, bg=BG)
frame.pack()
root.resizable(width=False, height=False)


def send():
    msg = message.get()
    message.delete(0, END)
    if msg != '':
        text_line.config(state=NORMAL)
        text_line.insert(END, 'You: '+msg+ '\n')
        bot_msg = chatbot_response(msg)
        text_line.insert(END, 'Bot: '+bot_msg+'\n\n')

        text_line.config(state=DISABLED)


header = Label(frame, text='Welcome From Py Chat', bg=BG, fg=TEXT, pady=10, font=50)
header.place(relwidth=1)
h_line = Label(frame, width=450, bg='#e8e8e8')
h_line.place(relwidth=1, rely=0.07, relheight=0.001)
text_line = Text(frame, width=20, height=2, bg=BG, fg=TEXT, padx=10, pady=5)
text_line.place(relheight=0.75, relwidth=1, rely=0.08)
text_line.config(cursor='arrow', state=DISABLED)
scroll = Scrollbar(frame)
scroll.place(relheight=0.8, relx=0.974)
scroll.config(command=text_line.yview)
bottom_label = Label(frame, bg='#525252', height=80)
bottom_label.place(relwidth=1, rely=0.825)
#message
message = Entry(bottom_label,bg='#292929', fg=TEXT)
message.place(relwidth=0.75, relheight=0.06, rely=0.008, relx=0.011)
message.focus()
sent_btn = Button(bottom_label, text='Send', font=20, width=20, fg=TEXT, bg='#d60000', command=send)
sent_btn.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

root.mainloop()