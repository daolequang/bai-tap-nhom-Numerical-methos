import math

# ============================================================
# DỮ LIỆU: diện tích nhà (sq ft) và giá bán (USD)
# ============================================================
x_data = [
    2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494,
    1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031,
    1767, 1888, 1604, 1962, 3890
]
y_data = [
    399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999,
    212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900,
    299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900, 573900
]
n = len(x_data)

# ============================================================
# HÀM GIẢI HỆ PHƯƠNG TRÌNH TUYẾN TÍNH (GAUSS ELIMINATION)
# Giải ma trận tăng cường [A|b] kích thước m x (m+1)
# ============================================================
def gauss_elimination(matrix):
    m = len(matrix)
    # Khử xuôi (Forward Elimination)
    for col in range(m):
        # Tìm hàng có phần tử lớn nhất (partial pivoting)
        max_row = col
        for row in range(col + 1, m):
            if abs(matrix[row][col]) > abs(matrix[max_row][col]):
                max_row = row
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]

        pivot = matrix[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Ma trận suy biến - hệ vô nghiệm hoặc vô số nghiệm")

        for row in range(col + 1, m):
            factor = matrix[row][col] / pivot
            for j in range(col, m + 1):
                matrix[row][j] -= factor * matrix[col][j]

    # Thế ngược (Back Substitution)
    solution = [0.0] * m
    for row in range(m - 1, -1, -1):
        solution[row] = matrix[row][m]
        for j in range(row + 1, m):
            solution[row] -= matrix[row][j] * solution[j]
        solution[row] /= matrix[row][row]

    return solution

# ============================================================
# HÀM TÍNH R² (Hệ số xác định)
# ============================================================
def r_squared(y_actual, y_predicted):
    y_mean = sum(y_actual) / len(y_actual)
    ss_tot = sum((y - y_mean) ** 2 for y in y_actual)
    ss_res = sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))
    return 1 - ss_res / ss_tot

# ============================================================
# MÔ HÌNH 1: HỒI QUY TUYẾN TÍNH  y = a0 + a1*x
# Hệ phương trình chuẩn:
#   n*a0    + Σx*a1    = Σy
#   Σx*a0   + Σx²*a1  = Σxy
# ============================================================
def linear_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))

    # Ma trận tăng cường [A|b]
    matrix = [
        [float(n),     sum_x,  sum_y],
        [sum_x,        sum_x2, sum_xy]
    ]

    a0, a1 = gauss_elimination(matrix)
    return a0, a1

# ============================================================
# MÔ HÌNH 2: HỒI QUY BẬC 2  y = a0 + a1*x + a2*x²
# Hệ phương trình chuẩn:
#   n*a0   + Σx*a1   + Σx²*a2  = Σy
#   Σx*a0  + Σx²*a1  + Σx³*a2  = Σxy
#   Σx²*a0 + Σx³*a1  + Σx⁴*a2  = Σx²y
# ============================================================
def quadratic_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))
    sum_x2y = sum(x[i]**2 * y[i] for i in range(n))

    # Ma trận tăng cường [A|b]
    matrix = [
        [float(n), sum_x,  sum_x2, sum_y],
        [sum_x,    sum_x2, sum_x3, sum_xy],
        [sum_x2,   sum_x3, sum_x4, sum_x2y]
    ]

    a0, a1, a2 = gauss_elimination(matrix)
    return a0, a1, a2

# ============================================================
# MÔ HÌNH 3: HỒI QUY HÀM MŨ  y = a * e^(b*x)
# Tuyến tính hoá: ln(y) = ln(a) + b*x
# Đặt Y = ln(y), A = ln(a) → Y = A + b*x (dạng tuyến tính)
# Hệ phương trình chuẩn:
#   n*A    + Σx*b   = ΣY
#   Σx*A   + Σx²*b  = ΣxY
# ============================================================
def exponential_regression(x, y, n):
    Y = [math.log(yi) for yi in y]   # Y = ln(y)

    sum_x  = sum(x)
    sum_Y  = sum(Y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xY = sum(x[i]*Y[i] for i in range(n))

    # Ma trận tăng cường [A|b]
    matrix = [
        [float(n), sum_x,  sum_Y],
        [sum_x,    sum_x2, sum_xY]
    ]

    A, b = gauss_elimination(matrix)
    a = math.exp(A)   # a = e^A
    return a, b

# ============================================================
# CHẠY 3 MÔ HÌNH VÀ IN KẾT QUẢ
# ============================================================

# --- Mô hình 1: Tuyến tính ---
a0, a1 = linear_regression(x_data, y_data, n)
y_pred_linear = [a0 + a1 * x for x in x_data]
r2_linear = r_squared(y_data, y_pred_linear)

print("=" * 55)
print("MÔ HÌNH 1: HỒI QUY TUYẾN TÍNH")
print("  y = a0 + a1 * x")
print(f"  a0 = {a0:,.4f}")
print(f"  a1 = {a1:,.4f}")
print(f"  Phương trình: y = {a0:,.2f} + {a1:,.4f} * x")
print(f"  R² = {r2_linear:.6f}")

# --- Mô hình 2: Bậc 2 ---
a0, a1, a2 = quadratic_regression(x_data, y_data, n)
y_pred_quad = [a0 + a1 * x + a2 * x**2 for x in x_data]
r2_quad = r_squared(y_data, y_pred_quad)

print("=" * 55)
print("MÔ HÌNH 2: HỒI QUY BẬC 2")
print("  y = a0 + a1*x + a2*x²")
print(f"  a0 = {a0:,.4f}")
print(f"  a1 = {a1:,.4f}")
print(f"  a2 = {a2:,.8f}")
print(f"  Phương trình: y = {a0:,.2f} + {a1:,.4f}*x + {a2:,.8f}*x²")
print(f"  R² = {r2_quad:.6f}")

# --- Mô hình 3: Hàm mũ ---
a, b = exponential_regression(x_data, y_data, n)
y_pred_exp = [a * math.exp(b * x) for x in x_data]
r2_exp = r_squared(y_data, y_pred_exp)

print("=" * 55)
print("MÔ HÌNH 3: HỒI QUY HÀM MŨ")
print("  y = a * e^(b*x)")
print(f"  a = {a:,.4f}")
print(f"  b = {b:,.8f}")
print(f"  Phương trình: y = {a:,.4f} * e^({b:,.8f} * x)")
print(f"  R² = {r2_exp:.6f}")

print("=" * 55)
print("SO SÁNH R² CÁC MÔ HÌNH:")
print(f"  Tuyến tính : R² = {r2_linear:.6f}")
print(f"  Bậc 2      : R² = {r2_quad:.6f}")
print(f"  Hàm mũ     : R² = {r2_exp:.6f}")
best = max([("Tuyến tính", r2_linear), ("Bậc 2", r2_quad), ("Hàm mũ", r2_exp)],
           key=lambda t: t[1])
print(f"  → Mô hình tốt nhất: {best[0]} (R² = {best[1]:.6f})")
print("=" * 55)
