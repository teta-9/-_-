import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool

def load_data():
    global input
    file_path = filedialog.askopenfilename(
        title="Выберите файл",
        filetypes=(("Таблицы", "*.csv"), ("Все файлы", "*.*"))
    )
    if file_path:  # Если файл выбран
        input = pd.read_csv(file_path)
        table.delete(*table.get_children())
        process_button.config(state=tk.NORMAL)
        lbl4.config(text="Данные загружены!") 
        lbl5.config(text="")
        lbl6.config(text="")
        save_button.config(state=tk.DISABLED)


def process_data():
    global results

    # предобработка
    input_ = input.fillna(0)
    input_.set_index('report_date', inplace=True)
    # Удаление столбцов не подходящих модели
    columns_to_keep = ['report_date']  # Исключаемый столбец
    columns_to_drop = [
        col for col in input_.select_dtypes(include='object').columns
        if col not in columns_to_keep
    ]
    input_ = input_.drop(columns=columns_to_drop)

    # Анализ данных
    predictions = model.predict(input_)

    # Результат
    client_ids = input['client_id']
    results = pd.DataFrame({
        'client_id': client_ids,
        'target': predictions
    })

    lbl5.config(text="Данные обработаны!")
    lbl6.config(text="Сохранить данные?")
    # Вывод в таблицу
    for row in table.get_children():
        table.delete(row)
    for _, row in results.iterrows():
        table.insert("", "end", values=(row['client_id'], row['target']))

    save_button.config(state=tk.NORMAL)


def save_data():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv")
    if file_path:
        results.to_csv(file_path, index=False)
        lbl6.config(text="Данные сохранены!")


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    


# Создаем главное окно
root = tk.Tk()
root.title("Определение склонности к приобретению машиноместа")

# Создание отступов
lbl1 = tk.Label(root)
lbl1.grid(row=4)

lbl2 = tk.Label(root)
lbl2.grid(row=0)

lbl3 = tk.Label(root)
lbl3.grid(row=2)


# Кнопки для загрузки, сохранения и обработки
load_button = tk.Button(root, text="Загрузить файл", command=load_data)
load_button.grid(row=1, column=0)


process_button = tk.Button(root, text="Обработать данные", command=process_data, state=tk.DISABLED)
process_button.grid(row=1, column=1)


save_button = tk.Button(root, text="Сохранить файл", command=save_data, state=tk.DISABLED)
save_button.grid(row=1, column=2)


# Создание полей для вывода статуса
lbl4 = tk.Label(root, text='Загрузите данные')
lbl4.grid(row=3, column=0)


lbl5 = tk.Label(root, text='')
lbl5.grid(row=3, column=1)

lbl6 = tk.Label(root, text='')
lbl6.grid(row=3, column=2)

# Создание таблицы
table = ttk.Treeview(root, columns=("Column1", "Column2"))
table.heading("Column1", text="client_id")
table.heading("Column2", text="target")
table.grid(row=5, column=0, columnspan=3)


root.mainloop()


