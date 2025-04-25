import sympy as sp

def gradient1(M_0, h, f, delta, minmax, update_table_callback, result_callback = None):
    x1, x2 = sp.symbols('x1 x2')
    func = sp.sympify(f)
    max_iter = 20

    optimization_path = []

    direction = 1 if (minmax == "max") else -1
    for k in range(max_iter):
        f_current = float(func.evalf(subs={x1: M_0[0], x2: M_0[1]}))
        optimization_path.append((M_0[0], M_0[1], f_current))
        if result_callback:
            result_callback(optimization_path)

        while True:  # Внутренний цикл для обработки уменьшения шага
            df_dx1 = float(sp.diff(func, x1).evalf(subs={x1: M_0[0], x2: M_0[1]}))
            df_dx2 = float(sp.diff(func, x2).evalf(subs={x1: M_0[0], x2: M_0[1]}))
            
            # Градиент
            grad = [
                df_dx1,
                df_dx2                
            ]
            
            # Обновляем позицию
            M_1 = [M_0[0] + direction * h * grad[0], 
                   M_0[1] + direction * h * grad[1]]
            
            f_M0 = float(func.evalf(subs={x1: M_0[0], x2: M_0[1]}))
            f_M1 = float(func.evalf(subs={x1:M_1[0], x2:M_1[1]}))
            
            # Проверяем условие для уменьшения шага
            if minmax == "max" and f_M1 < f_M0 or minmax == "min" and f_M1 > f_M0:
                h /= 2
                continue  # Повторяем итерацию с уменьшенным шагом
            else:
                break  # Выходим из внутреннего цикла
        
        optimization_path.append((M_1[0], M_1[1], f_M1))

        # Обновляем таблицу
        update_table_callback(
            k, h,
            f"({M_0[0]:.2f}, {M_0[1]:.2f})",
            f"({grad[0]:.2f}, {grad[1]:.2f})",
            f_M0, f_M1
        )
        
        # Проверяем условие сходимости
        if (abs(f_M1 - f_M0) <= delta):
            return M_1, f_M1
        elif ((minmax == "max" and f_M1 > f_M0) or (minmax == "min" and f_M1 < f_M0)):
            M_0 = M_1
            continue
    return M_0, f_M0

def conGradient(M_0, f, delta, minmax, update_table_callback, result_callback):
    x1, x2 = sp.symbols('x1 x2')
    h = sp.symbols('h')  # Символ для шага оптимизации
    func = sp.sympify(f)
    k = 0
    gradPrev = None
    directionPrev = None
    max_iter = 100
    
    optimization_path = []  
    # Определяем направление оптимизации
    direction = -1 if minmax == "min" else 1

    f_current = float(func.evalf(subs={x1: M_0[0], x2: M_0[1]}))
    optimization_path.append((M_0[0], M_0[1], f_current))
    if result_callback:
        result_callback(optimization_path)

    while k < max_iter:
        # Вычисляем текущий градиент
        df_dx1 = float(sp.diff(func, x1).evalf(subs={x1: M_0[0], x2: M_0[1]}))
        df_dx2 = float(sp.diff(func, x2).evalf(subs={x1: M_0[0], x2: M_0[1]}))
        grad = [df_dx1, df_dx2]

        # Определяем направление поиска
        if k == 0:
            d = [direction * g for g in grad]  # Первая итерация - градиент/антиградиент
        else:
            # Метод Флетчера-Ривса
            beta = sum(g**2 for g in grad) / sum(g**2 for g in gradPrev)
            d = [direction * g + beta * d_prev for g, d_prev in zip(grad, directionPrev)]

        # Строим функцию Z(h) = f(M_0 + h*d)
        Z = func.subs({
            x1: M_0[0] + h * d[0],
            x2: M_0[1] + h * d[1]
        })

        # Решаем Z'(h) = 0 аналитически
        dZ_dh = sp.diff(Z, h)
        try:
            h_solutions = sp.solve(dZ_dh, h)
            # Выбираем действительные решения
            real_solutions = [sol.evalf() for sol in h_solutions if sol.is_real]
            
            if not real_solutions:
                raise ValueError("No real solutions for h")
                
            # Для каждого решения вычисляем Z(h) и выбираем оптимальное
            h_values = []
            for sol in real_solutions:
                h_val = float(sol)
                Z_val = Z.subs(h, h_val)
                h_values.append((h_val, Z_val))
            
            # Выбираем h в зависимости от задачи (max/min)
            if minmax == "max":
                h_opt, _ = max(h_values, key=lambda x: x[1])
            else:
                h_opt, _ = min(h_values, key=lambda x: x[1])
                
        except Exception as e:
            print(f"Warning: {e}, using default h=0.1")
            h_opt = 0.1

        # Новое положение
        M_1 = [
            M_0[0] + h_opt * d[0],
            M_0[1] + h_opt * d[1]
        ]

        # Вычисляем значения функции
        f_M0 = float(func.evalf(subs={x1: M_0[0], x2: M_0[1]}))
        f_M1 = float(func.evalf(subs={x1: M_1[0], x2: M_1[1]}))

        # Добавляем новую точку в путь оптимизации
        optimization_path.append((M_1[0], M_1[1], f_M1))
        if result_callback:
            result_callback(optimization_path)
        
        # Обновляем таблицу
        update_table_callback(
            k, round(h_opt, 3),  # Точность до 6 знаков
            f"({M_0[0]:.3f}, {M_0[1]:.3f})",
            f"({grad[0]:.3f}, {grad[1]:.3f})",
            round(f_M0, 3),
            round(f_M1, 3)
        )

        # Проверка условия остановки
        if abs(f_M1 - f_M0) <= delta:
            return M_1, f_M1

        # Подготовка к следующей итерации
        elif ((minmax == "max" and f_M1 > f_M0) or (minmax == "min" and f_M1 < f_M0)):
            M_0 = M_1
            gradPrev = grad
            directionPrev = d
            k += 1
            continue
        else: break
    return M_0, f_M0
