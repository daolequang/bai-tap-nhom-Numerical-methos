import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 1. DỮ LIỆU THỰC TẾ (30 điểm dữ liệu tối ưu)
# ============================================================
x_data = [
    2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494,
    1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031,
    1767, 1888, 1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637
]
y_data = [
    399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999,
    212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900,
    299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900,
    573900, 249900, 464500, 469000, 475000, 299900
]
n = len(x_data)

# ============================================================
# 2. XÂY DỰNG THUẬT TOÁN GIẢI HỆ PHƯƠNG TRÌNH CHUẨN (GAUSS)
# Yêu cầu cốt lõi: Tự xây dựng thuật toán tìm hệ số
# ============================================================
def gauss_elimination(matrix):
    m = len(matrix)
    # Khử xuôi (Forward Elimination)
    for col in range(m):
        # Partial pivoting
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

# Hàm tính MSE và RMSE theo yêu cầu đề bài
def calculate_errors(y_true, y_pred):
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)
    return mse, math.sqrt(mse)

def calculate_r_squared(y_true, y_pred):
    ss_total = sum((y - np.mean(y_true)) ** 2 for y in y_true)
    ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    if ss_total == 0:
        return 0.0 # Avoid division by zero if all y_true values are the same
    return 1 - (ss_residual / ss_total)

# ============================================================
# 3. TÍNH TOÁN CÁC MÔ HÌNH HỒI QUY
# ============================================================
# --- MÔ HÌNH 1: HỒI QUY TUYẾN TÍNH (y = a0 + a1*x) ---
def linear_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))

    matrix = [
        [float(n), sum_x, sum_y],
        [sum_x, sum_x2, sum_xy]
    ]
    return gauss_elimination(matrix)

# --- MÔ HÌNH 2: HỒI QUY BẬC 2 (y = a0 + a1*x + a2*x²) ---
def quadratic_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))
    sum_x2y = sum(x[i]**2 * y[i] for i in range(n))

    matrix = [
        [float(n), sum_x,  sum_x2, sum_y],
        [sum_x,    sum_x2, sum_x3, sum_xy],
        [sum_x2,   sum_x3, sum_x4, sum_x2y]
    ]
    return gauss_elimination(matrix)

# --- MÔ HÌNH 3: HỒI QUY HÀM MŨ (y = a * e^(b*x)) ---
def exponential_regression(x, y, n):
    Y = [math.log(yi) for yi in y]
    sum_x  = sum(x)
    sum_Y  = sum(Y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xY = sum(x[i]*Y[i] for i in range(n))

    matrix = [
        [float(n), sum_x, sum_Y],
        [sum_x, sum_x2, sum_xY]
    ]
    A, b = gauss_elimination(matrix)
    a = math.exp(A)
    return a, b

# ============================================================
# 4. CHẠY MÔ HÌNH VÀ IN BẢNG KẾT QUẢ ĐÁNH GIÁ
# ============================================================
a0_lin, a1_lin = linear_regression(x_data, y_data, n)
y_pred_linear = [a0_lin + a1_lin * x for x in x_data]
mse_lin, rmse_lin = calculate_errors(y_data, y_pred_linear)
r2_linear = calculate_r_squared(y_data, y_pred_linear)

a0_q, a1_q, a2_q = quadratic_regression(x_data, y_data, n)
y_pred_quad = [a0_q + a1_q * x + a2_q * x**2 for x in x_data]
mse_quad, rmse_quad = calculate_errors(y_data, y_pred_quad)
r2_quad = calculate_r_squared(y_data, y_pred_quad)

a_exp, b_exp = exponential_regression(x_data, y_data, n)
y_pred_exp = [a_exp * math.exp(b_exp * x) for x in x_data]
mse_exp, rmse_exp = calculate_errors(y_data, y_pred_exp)
r2_exp = calculate_r_squared(y_data, y_pred_exp)

print("=" * 55)
print("MÔ HÌNH 1: HỒI QUY TUYẾN TÍNH")
print("  y = a0 + a1 * x")
print(f"  a0 = {a0_lin:,.4f}")
print(f"  a1 = {a1_lin:,.4f}")
print(f"  Phương trình: y = {a0_lin:,.2f} + {a1_lin:,.4f} * x")
print(f"  R² = {r2_linear:.6f}")
print("=" * 55)
print("MÔ HÌNH 2: HỒI QUY BẬC 2")
print("  y = a0 + a1*x + a2*x²")
print(f"  a0 = {a0_q:,.4f}")
print(f"  a1 = {a1_q:,.4f}")
print(f"  a2 = {a2_q:,.8f}")
print(f"  Phương trình: y = {a0_q:,.2f} + {a1_q:,.4f}*x + {a2_q:,.8f}*x²")
print(f"  R² = {r2_quad:.6f}")
print("=" * 55)
print("MÔ HÌNH 3: HỒI QUY HÀM MŨ")
print("  y = a * e^(b*x)")
print(f"  a = {a_exp:,.4f}")
print(f"  b = {b_exp:,.8f}")
print(f"  Phương trình: y = {a_exp:,.4f} * e^({b_exp:,.8f} * x)")
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

print("\n" + "=" * 65)
print("BẢNG SO SÁNH SAI SỐ GIỮA CÁC MÔ HÌNH (Dữ liệu 30 điểm)")
print("=" * 65)
print(f"{'Mô hình':<25} | {'MSE':>18} | {'RMSE (USD)':>15}")
print("-" * 65)
print(f"{'Tuyến tính (y = a0 + a1*x)':<25} | {mse_lin:>18,.2f} | {rmse_lin:>15,.2f}")
print(f"{'Đa thức Bậc 2':<25} | {mse_quad:>18,.2f} | {rmse_quad:>15,.2f}")
print(f"{'Hàm mũ (y = a*e^(bx))':<25} | {mse_exp:>18,.2f} | {rmse_exp:>15,.2f}")
print("=" * 65)

# Đánh giá tự động
models_eval = [
    ("Tuyến tính", rmse_lin),
    ("Đa thức Bậc 2", rmse_quad),
    ("Hàm mũ", rmse_exp)
]
models_eval.sort(key=lambda t: t[1])
best_model, best_rmse = models_eval[0]
second_best_model, second_rmse = models_eval[1]

print(f"\nKẾT LUẬN:")
print(f"Mô hình biểu diễn dữ liệu tốt nhất là [{best_model}] với RMSE nhỏ nhất: {best_rmse:,.2f} USD")

diff_pct = (second_rmse - best_rmse) / best_rmse * 100
if diff_pct < 1.0 and best_model != "Tuyến tính":
    print(f"Tuy nhiên, sự cải thiện so với mô hình [{second_best_model}] chỉ là {diff_pct:.2f} %.")
    print(f"Khuyến nghị: Nếu ưu tiên tính toán nhẹ và giải thích dễ dàng, có thể cân nhắc dùng mô hình đơn giản hơn.")

# ============================================================
# 5. TRỰC QUAN HÓA BẰNG MATPLOTLIB (Lấy từ code Toàn)
# ============================================================
X_np = np.array(x_data)
y_np = np.array(y_data)
x_line = np.linspace(X_np.min(), X_np.max(), 400)

y_line_lin  = a0_lin + a1_lin * x_line
y_line_quad = a0_q   + a1_q * x_line + a2_q * x_line**2
y_line_exp  = a_exp  * np.exp(b_exp * x_line)

COLORS = {"lin": "#2563EB", "quad": "#16A34A", "exp": "#DC2626"}

fig = plt.figure(figsize=(14, 9), facecolor="#F8FAFC")
fig.suptitle("Đánh giá mô hình hồi quy - Diện tích nhà (sq ft) và Giá bán (USD)",
             fontsize=14, fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35, top=0.88, bottom=0.08, left=0.07, right=0.97)
fmt = plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K")

subplots_cfg = [
    (gs[0,0], "Tuyến tính (Bậc 1)", y_line_lin, COLORS["lin"], f"y={a0_lin:,.0f}+{a1_lin:.2f}x", rmse_lin),
    (gs[0,1], "Đa thức (Bậc 2)", y_line_quad, COLORS["quad"], f"y={a0_q:,.0f}+{a1_q:.2f}x...", rmse_quad),
    (gs[0,2], "Hàm mũ", y_line_exp, COLORS["exp"], f"y={a_exp:.0f}·e^({b_exp:.5f}x)", rmse_exp),
]

for spec, title, yl, color, eq_lbl, rmse_v in subplots_cfg:
    ax = fig.add_subplot(spec)
    ax.scatter(X_np, y_np, color="#64748B", s=40, zorder=5, label="Dữ liệu")
    ax.plot(x_line, yl, color=color, lw=2.2, label=f"{eq_lbl}\nRMSE={rmse_v:,.0f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Diện tích (sq ft)")
    ax.set_ylabel("Giá bán (USD)")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

ax_all = fig.add_subplot(gs[1, 0:2])
ax_all.scatter(X_np, y_np, color="#64748B", s=50, zorder=5, label="Dữ liệu thực tế")
ax_all.plot(x_line, y_line_lin, color=COLORS["lin"], lw=2, ls="--", label=f"Tuyến tính (RMSE={rmse_lin:,.0f})")
ax_all.plot(x_line, y_line_quad, color=COLORS["quad"], lw=2.5, ls="-", label=f"Bậc 2 (RMSE={rmse_quad:,.0f})")
ax_all.plot(x_line, y_line_exp, color=COLORS["exp"], lw=2, ls="-.", label=f"Hàm mũ (RMSE={rmse_exp:,.0f})")
ax_all.set_title("So sánh trực quan các mô hình", fontweight="bold")
ax_all.set_xlabel("Diện tích (sq ft)")
ax_all.set_ylabel("Giá bán (USD)")
ax_all.yaxis.set_major_formatter(fmt)
ax_all.legend(fontsize=9)
ax_all.grid(alpha=0.3)

ax_bar = fig.add_subplot(gs[1, 2])
xpos = np.arange(3)
w = 0.4
rmse_list = [rmse_lin, rmse_quad, rmse_exp]
bar_colors = [COLORS["lin"], COLORS["quad"], COLORS["exp"]]

b2b = ax_bar.bar(xpos, [v/1e3 for v in rmse_list], w, label="RMSE (×10³ USD)", color=bar_colors, alpha=0.9, hatch="//")
for bar in b2b:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f"{h:.1f}K", ha="center", va="bottom", fontsize=8)

ax_bar.set_xticks(xpos)
ax_bar.set_xticklabels(["Tuyến tính", "Bậc 2", "Hàm mũ"], fontsize=9)
ax_bar.set_title("So sánh RMSE", fontweight="bold")
ax_bar.grid(axis='y', alpha=0.3)

plt.savefig("danh_gia_mo_hinh.png", dpi=150, bbox_inches="tight")
plt.show()import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 1. DỮ LIỆU THỰC TẾ (30 điểm dữ liệu tối ưu)
# ============================================================
x_data = [
    2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494,
    1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031,
    1767, 1888, 1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637
]
y_data = [
    399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999,
    212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900,
    299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900,
    573900, 249900, 464500, 469000, 475000, 299900
]
n = len(x_data)

# ============================================================
# 2. XÂY DỰNG THUẬT TOÁN GIẢI HỆ PHƯƠNG TRÌNH CHUẨN (GAUSS)
# Yêu cầu cốt lõi: Tự xây dựng thuật toán tìm hệ số
# ============================================================
def gauss_elimination(matrix):
    m = len(matrix)
    # Khử xuôi (Forward Elimination)
    for col in range(m):
        # Partial pivoting
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

# Hàm tính MSE và RMSE theo yêu cầu đề bài
def calculate_errors(y_true, y_pred):
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)
    return mse, math.sqrt(mse)

def calculate_r_squared(y_true, y_pred):
    ss_total = sum((y - np.mean(y_true)) ** 2 for y in y_true)
    ss_residual = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    if ss_total == 0:
        return 0.0 # Avoid division by zero if all y_true values are the same
    return 1 - (ss_residual / ss_total)

# ============================================================
# 3. TÍNH TOÁN CÁC MÔ HÌNH HỒI QUY
# ============================================================
# --- MÔ HÌNH 1: HỒI QUY TUYẾN TÍNH (y = a0 + a1*x) ---
def linear_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))

    matrix = [
        [float(n), sum_x, sum_y],
        [sum_x, sum_x2, sum_xy]
    ]
    return gauss_elimination(matrix)

# --- MÔ HÌNH 2: HỒI QUY BẬC 2 (y = a0 + a1*x + a2*x²) ---
def quadratic_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))
    sum_x2y = sum(x[i]**2 * y[i] for i in range(n))

    matrix = [
        [float(n), sum_x,  sum_x2, sum_y],
        [sum_x,    sum_x2, sum_x3, sum_xy],
        [sum_x2,   sum_x3, sum_x4, sum_x2y]
    ]
    return gauss_elimination(matrix)

# --- MÔ HÌNH 3: HỒI QUY HÀM MŨ (y = a * e^(b*x)) ---
def exponential_regression(x, y, n):
    Y = [math.log(yi) for yi in y]
    sum_x  = sum(x)
    sum_Y  = sum(Y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xY = sum(x[i]*Y[i] for i in range(n))

    matrix = [
        [float(n), sum_x, sum_Y],
        [sum_x, sum_x2, sum_xY]
    ]
    A, b = gauss_elimination(matrix)
    a = math.exp(A)
    return a, b

# ============================================================
# 4. CHẠY MÔ HÌNH VÀ IN BẢNG KẾT QUẢ ĐÁNH GIÁ
# ============================================================
a0_lin, a1_lin = linear_regression(x_data, y_data, n)
y_pred_linear = [a0_lin + a1_lin * x for x in x_data]
mse_lin, rmse_lin = calculate_errors(y_data, y_pred_linear)
r2_linear = calculate_r_squared(y_data, y_pred_linear)

a0_q, a1_q, a2_q = quadratic_regression(x_data, y_data, n)
y_pred_quad = [a0_q + a1_q * x + a2_q * x**2 for x in x_data]
mse_quad, rmse_quad = calculate_errors(y_data, y_pred_quad)
r2_quad = calculate_r_squared(y_data, y_pred_quad)

a_exp, b_exp = exponential_regression(x_data, y_data, n)
y_pred_exp = [a_exp * math.exp(b_exp * x) for x in x_data]
mse_exp, rmse_exp = calculate_errors(y_data, y_pred_exp)
r2_exp = calculate_r_squared(y_data, y_pred_exp)

print("=" * 55)
print("MÔ HÌNH 1: HỒI QUY TUYẾN TÍNH")
print("  y = a0 + a1 * x")
print(f"  a0 = {a0_lin:,.4f}")
print(f"  a1 = {a1_lin:,.4f}")
print(f"  Phương trình: y = {a0_lin:,.2f} + {a1_lin:,.4f} * x")
print(f"  R² = {r2_linear:.6f}")
print("=" * 55)
print("MÔ HÌNH 2: HỒI QUY BẬC 2")
print("  y = a0 + a1*x + a2*x²")
print(f"  a0 = {a0_q:,.4f}")
print(f"  a1 = {a1_q:,.4f}")
print(f"  a2 = {a2_q:,.8f}")
print(f"  Phương trình: y = {a0_q:,.2f} + {a1_q:,.4f}*x + {a2_q:,.8f}*x²")
print(f"  R² = {r2_quad:.6f}")
print("=" * 55)
print("MÔ HÌNH 3: HỒI QUY HÀM MŨ")
print("  y = a * e^(b*x)")
print(f"  a = {a_exp:,.4f}")
print(f"  b = {b_exp:,.8f}")
print(f"  Phương trình: y = {a_exp:,.4f} * e^({b_exp:,.8f} * x)")
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

print("\n" + "=" * 65)
print("BẢNG SO SÁNH SAI SỐ GIỮA CÁC MÔ HÌNH (Dữ liệu 30 điểm)")
print("=" * 65)
print(f"{'Mô hình':<25} | {'MSE':>18} | {'RMSE (USD)':>15}")
print("-" * 65)
print(f"{'Tuyến tính (y = a0 + a1*x)':<25} | {mse_lin:>18,.2f} | {rmse_lin:>15,.2f}")
print(f"{'Đa thức Bậc 2':<25} | {mse_quad:>18,.2f} | {rmse_quad:>15,.2f}")
print(f"{'Hàm mũ (y = a*e^(bx))':<25} | {mse_exp:>18,.2f} | {rmse_exp:>15,.2f}")
print("=" * 65)

# Đánh giá tự động
models_eval = [
    ("Tuyến tính", rmse_lin),
    ("Đa thức Bậc 2", rmse_quad),
    ("Hàm mũ", rmse_exp)
]
models_eval.sort(key=lambda t: t[1])
best_model, best_rmse = models_eval[0]
second_best_model, second_rmse = models_eval[1]

print(f"\nKẾT LUẬN:")
print(f"Mô hình biểu diễn dữ liệu tốt nhất là [{best_model}] với RMSE nhỏ nhất: {best_rmse:,.2f} USD")

diff_pct = (second_rmse - best_rmse) / best_rmse * 100
if diff_pct < 1.0 and best_model != "Tuyến tính":
    print(f"Tuy nhiên, sự cải thiện so với mô hình [{second_best_model}] chỉ là {diff_pct:.2f} %.")
    print(f"Khuyến nghị: Nếu ưu tiên tính toán nhẹ và giải thích dễ dàng, có thể cân nhắc dùng mô hình đơn giản hơn.")

# ============================================================
# 5. TRỰC QUAN HÓA BẰNG MATPLOTLIB (Lấy từ code Toàn)
# ============================================================
X_np = np.array(x_data)
y_np = np.array(y_data)
x_line = np.linspace(X_np.min(), X_np.max(), 400)

y_line_lin  = a0_lin + a1_lin * x_line
y_line_quad = a0_q   + a1_q * x_line + a2_q * x_line**2
y_line_exp  = a_exp  * np.exp(b_exp * x_line)

COLORS = {"lin": "#2563EB", "quad": "#16A34A", "exp": "#DC2626"}

fig = plt.figure(figsize=(14, 9), facecolor="#F8FAFC")
fig.suptitle("Đánh giá mô hình hồi quy - Diện tích nhà (sq ft) và Giá bán (USD)",
             fontsize=14, fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35, top=0.88, bottom=0.08, left=0.07, right=0.97)
fmt = plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K")

subplots_cfg = [
    (gs[0,0], "Tuyến tính (Bậc 1)", y_line_lin, COLORS["lin"], f"y={a0_lin:,.0f}+{a1_lin:.2f}x", rmse_lin),
    (gs[0,1], "Đa thức (Bậc 2)", y_line_quad, COLORS["quad"], f"y={a0_q:,.0f}+{a1_q:.2f}x...", rmse_quad),
    (gs[0,2], "Hàm mũ", y_line_exp, COLORS["exp"], f"y={a_exp:.0f}·e^({b_exp:.5f}x)", rmse_exp),
]

for spec, title, yl, color, eq_lbl, rmse_v in subplots_cfg:
    ax = fig.add_subplot(spec)
    ax.scatter(X_np, y_np, color="#64748B", s=40, zorder=5, label="Dữ liệu")
    ax.plot(x_line, yl, color=color, lw=2.2, label=f"{eq_lbl}\nRMSE={rmse_v:,.0f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Diện tích (sq ft)")
    ax.set_ylabel("Giá bán (USD)")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

ax_all = fig.add_subplot(gs[1, 0:2])
ax_all.scatter(X_np, y_np, color="#64748B", s=50, zorder=5, label="Dữ liệu thực tế")
ax_all.plot(x_line, y_line_lin, color=COLORS["lin"], lw=2, ls="--", label=f"Tuyến tính (RMSE={rmse_lin:,.0f})")
ax_all.plot(x_line, y_line_quad, color=COLORS["quad"], lw=2.5, ls="-", label=f"Bậc 2 (RMSE={rmse_quad:,.0f})")
ax_all.plot(x_line, y_line_exp, color=COLORS["exp"], lw=2, ls="-.", label=f"Hàm mũ (RMSE={rmse_exp:,.0f})")
ax_all.set_title("So sánh trực quan các mô hình", fontweight="bold")
ax_all.set_xlabel("Diện tích (sq ft)")
ax_all.set_ylabel("Giá bán (USD)")
ax_all.yaxis.set_major_formatter(fmt)
ax_all.legend(fontsize=9)
ax_all.grid(alpha=0.3)

ax_bar = fig.add_subplot(gs[1, 2])
xpos = np.arange(3)
w = 0.4
rmse_list = [rmse_lin, rmse_quad, rmse_exp]
bar_colors = [COLORS["lin"], COLORS["quad"], COLORS["exp"]]

b2b = ax_bar.bar(xpos, [v/1e3 for v in rmse_list], w, label="RMSE (×10³ USD)", color=bar_colors, alpha=0.9, hatch="//")
for bar in b2b:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f"{h:.1f}K", ha="center", va="bottom", fontsize=8)

ax_bar.set_xticks(xpos)
ax_bar.set_xticklabels(["Tuyến tính", "Bậc 2", "Hàm mũ"], fontsize=9)
ax_bar.set_title("So sánh RMSE", fontweight="bold")
ax_bar.grid(axis='y', alpha=0.3)

plt.savefig("danh_gia_mo_hinh.png", dpi=150, bbox_inches="tight")
plt.show()
