import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox
import numpy as np
import optimizerV2
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

window = tk.Tk()
window.title("Функциональный оптимизатор")
window.geometry("1000x600")

# Основные фреймы
main_frame = tk.Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

left_frame = tk.Frame(main_frame)
left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

right_frame = tk.Frame(main_frame)
right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=3)
main_frame.grid_rowconfigure(0, weight=1)

# Элементы управления
var = tk.StringVar()
var_minmax = tk.StringVar()
var_minmax.set("min")

# Поля ввода
tk.Label(left_frame, text="Enter your function:").grid(row=0, column=1)
function_entry = Entry(left_frame)
function_entry.grid(row=1, column=1)

tk.Label(left_frame, text="Initial point:").grid(row=2, column=1)
initial_point_frame = tk.Frame(left_frame)
initial_point_frame.grid(row=3, column=1, sticky="ew")

tk.Label(initial_point_frame, text="x1:").pack(side=LEFT)
x1_entry = Entry(initial_point_frame, width=10)
x1_entry.pack(side=LEFT, padx=5)

tk.Label(initial_point_frame, text="x2:").pack(side=LEFT)
x2_entry = Entry(initial_point_frame, width=10)
x2_entry.pack(side=LEFT, padx=5)

step_label = tk.Label(left_frame, text="Step (h):")
step_label.grid(row=4, column=1)
step_entry = Entry(left_frame)
step_entry.grid(row=5, column=1)

# Новое поле для delta (δ)
tk.Label(left_frame, text="Delta (δ):").grid(row=6, column=1)
delta_entry = Entry(left_frame)
delta_entry.grid(row=7, column=1)

tk.Label(left_frame, text="Choose algorithm:").grid(row=8, column=1)
combo_method = Combobox(left_frame, textvariable=var, 
                       values=('Gradient method (1st order)', 'Conjugate gradient method'))
combo_method.current(0)
combo_method.grid(row=9, column=1)

def toggle_step_visibility(event=None):
    if combo_method.get() == "Conjugate gradient method":
        step_label.grid_remove()  # Скрываем надпись "Step (h):"
        step_entry.grid_remove()  # Скрываем поле ввода
    else:
        step_label.grid(row=4, column=1)  # Показываем надпись
        step_entry.grid(row=5, column=1)  # Показываем поле ввода

# Привязываем функцию к изменению выбора в комбобоксе
combo_method.bind("<<ComboboxSelected>>", toggle_step_visibility)

# Инициализируем видимость (скрываем, если изначально выбран Conjugate gradient)
toggle_step_visibility()

tk.Label(left_frame, text="Min or max:").grid(row=10, column=1)
combo_minmax = Combobox(left_frame, textvariable=var_minmax, values=('min', 'max'))
combo_minmax.current(0)
combo_minmax.grid(row=11, column=1)

solve = Button(left_frame, text='Solve')
solve.grid(row=12, column=1, pady=10)

# Таблица результатов
class CustomTable:
    def __init__(self, parent):
        self.parent = parent
        self.table_frame = tk.Frame(parent)
        self.table_frame.pack(fill="both", expand=True)
        self.rows = []
        self.columns = ['k', 'h', 'M_k', '∇f(M_k)', 'f(M_k)', 'f(M_k1)']
        self.create_header()

    def create_header(self):
        header_frame = tk.Frame(self.table_frame)
        header_frame.pack(fill="x")
        for col in self.columns:
            tk.Label(header_frame, text=col, borderwidth=1, relief="solid", width=10).pack(side="left")

    def add_row(self, data):
        row_frame = tk.Frame(self.table_frame)
        row_frame.pack(fill="x")
        for value in data:
            tk.Label(row_frame, text=str(value), borderwidth=1, relief="solid", width=10).pack(side="left")
        self.rows.append(row_frame)

    def clear_table(self):
        for row in self.rows:
            row.destroy()
        self.rows = []

table_header = tk.Label(right_frame, text="", font=("Arial", 14, "bold"))
table_header.grid(row=0, columnspan=2)

table_frame = Frame(right_frame)
table_frame.grid(row=1, columnspan=2, sticky="nsew")
right_frame.grid_rowconfigure(1, weight=1)

custom_table = CustomTable(table_frame)

def update_table(k, h, M_k, grad, f_M_k, f_M_k1):
    custom_table.add_row([k, h, M_k, grad, f"{float(f_M_k):.6f}", f"{float(f_M_k1):.6f}"])

# График
graph_frame = tk.Frame(right_frame)
graph_frame.grid(row=2, columnspan=2, sticky="nsew")
right_frame.grid_rowconfigure(2, weight=1)

def plot_function(f, points_history=None, func_type="", min_max=""):
    x1, x2 = sp.symbols('x1 x2')
    func = sp.lambdify((x1, x2), sp.sympify(f), 'numpy')
    
    # Определяем диапазон графика на основе точек оптимизации
    if points_history:
        x1_points = [p[0] for p in points_history]
        x2_points = [p[1] for p in points_history]
        
        # Добавляем отступы к диапазону
        x1_padding = max(1, (max(x1_points) - min(x1_points)) * 0.5)
        x2_padding = max(1, (max(x2_points) - min(x2_points)) * 0.5)
        
        x1_range = (min(x1_points) - x1_padding, max(x1_points) + x1_padding)
        x2_range = (min(x2_points) - x2_padding, max(x2_points) + x2_padding)
    else:
        x1_range = (-10, 10)
        x2_range = (-10, 10)

    # Создаем сетку для графика
    x1_vals = np.linspace(x1_range[0], x1_range[1], 100)
    x2_vals = np.linspace(x2_range[0], x2_range[1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = func(X1, X2)

    # Создаем фигуру
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Поверхность функции
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.5)
    
    # Добавляем точки оптимизации
    if points_history:
        x1_pts = [p[0] for p in points_history]
        x2_pts = [p[1] for p in points_history]
        z_pts = [p[2] for p in points_history]
        
        # Линия, соединяющая точки
        ax.plot(x1_pts, x2_pts, z_pts, 'r-', linewidth=2, marker='o', markersize=5)
        
        # Подписи для важных точек
        ax.text(x1_pts[0], x2_pts[0], z_pts[0], ' Start', color='green')
        ax.text(x1_pts[-1], x2_pts[-1], z_pts[-1], ' End', color='red')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title(f'Optimization Path\nMethod: {func_type} ({min_max})')

    return fig

optimization_path = []

def result_callback(path):
    global optimization_path
    optimization_path = path

def solve_function():
    try:
        custom_table.clear_table()
        func_type = combo_method.get()
        min_max = combo_minmax.get()
        f = function_entry.get()
        h = float(step_entry.get()) if func_type == "Gradient method (1st order)" else None
        delta = float(delta_entry.get())  # Получаем значение delta

        x1 = float(x1_entry.get())
        x2 = float(x2_entry.get())
        initial_point = [x1, x2]
        
        if not f:
            raise ValueError("Функция не введена")
            
    except ValueError as e:
        messagebox.showerror("Ошибка", f"Некорректные входные данные: {str(e)}")
        return

    table_header['text'] = f"Optimization result by method: {func_type} ({min_max})"

        # Очищаем предыдущий график
    for widget in graph_frame.winfo_children():
        widget.destroy()

        # Создаем начальный график
    fig = plot_function(f, func_type=func_type, min_max=min_max)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Создаем фрейм для результата
    result_frame = tk.Frame(right_frame)
    result_frame.grid(row=3, columnspan=2, sticky="ew", pady=10)
    
    # Очищаем предыдущий результат
    for widget in result_frame.winfo_children():
        widget.destroy()

    def result_callback(path):
        # Обновляем график с новыми точками
        for widget in graph_frame.winfo_children():
            widget.destroy()
        
        fig = plot_function(f, points_history=path, func_type=func_type, min_max=min_max)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        window.update()  # Обновляем окно

    if func_type == "Gradient method (1st order)":
        final_point, final_value = optimizerV2.gradient1(
            initial_point, h, f, delta, min_max, update_table, result_callback
        )
    elif func_type == "Conjugate gradient method":
        final_point, final_value = optimizerV2.conGradient(
            initial_point, f, delta, min_max, update_table, result_callback
        )
    result_label = tk.Label(
        result_frame,
        text=f"{'Maximum' if min_max == 'max' else 'Minimum'} point: ({final_point[0]:.6f}, {final_point[1]:.6f}), "
             f"Function value: {final_value:.6f}",
        font=("Arial", 10, "bold"),
        fg="green"
    )
    result_label.pack()    

solve.config(command=solve_function)

window.mainloop()