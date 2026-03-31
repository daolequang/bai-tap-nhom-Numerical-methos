"""
Toàn - Phần 3
Dữ liệu thực tế: Diện tích nhà (sq ft) và Giá bán (USD)
Giải hệ phương trình chuẩn bằng MA TRẬN
STT	Diện tích nhà (sq ft)	Giá bán (USD)
1	2104	399,900
2	1600	329,900
3	2400	369,000
4	1416	232,000
5	3000	539,900
6	1985	299,900
7	1534	314,900
8	1427	198,999
9	1380	212,000
10	1494	242,500
11	1940	239,999
12	2000	347,000
13	1890	329,999
14	4478	699,900
15	1268	259,900
16	2300	449,900
17	1320	299,900
18	1236	199,900
19	2609	499,998
20	3031	599,000
21	1767	252,900
22	1888	255,000
23	1604	242,900
24	1962	259,900
25	3890	573,900
26	1100	249,900
27	1458	464,500
28	2526	469,000
29	2200	475,000
30	2637	299,900
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Bước 1: Khởi tạo dữ liệu thực tế
X = np.array([2104,1600,2400,1416,3000,1985,1534,1427,1380,1494,
              1940,2000,1890,4478,1268,2300,1320,1236,2609,3031,
              1767,1888,1604,1962,3890,1100,1458,2526,2200,2637], dtype=float)

y = np.array([399900,329900,369000,232000,539900,299900,314900,198999,
              212000,242500,239999,347000,329999,699900,259900,449900,
              299900,199900,499998,599000,252900,255000,242900,259900,
              573900,249900,464500,469000,475000,299900], dtype=float)

n = len(X)
X_col = X.reshape(-1, 1)
y_col = y.reshape(-1, 1)

# Bước 2: Hàm giải Hệ phương trình chuẩn
def normal_equation(X_matrix, y_vector):
    """Giải (X^T X)·theta = X^T·y bằng np.linalg.solve (ổn định hơn inv)"""
    A = X_matrix.T.dot(X_matrix)
    b = X_matrix.T.dot(y_vector)
    return np.linalg.solve(A, b)

# Bước 3: Huấn luyện 3 mô hình
# Mô hình 1: Tuyến tính
X_linear = np.c_[np.ones(n), X_col]
theta_linear = normal_equation(X_linear, y_col)
y_pred_linear = X_linear.dot(theta_linear)

# Mô hình 2: Bậc 2
X_quad = np.c_[np.ones(n), X_col, X_col**2]
theta_quad = normal_equation(X_quad, y_col)
y_pred_quad = X_quad.dot(theta_quad)

# Mô hình 3: Hàm mũ (tuyến tính hóa qua ln)
y_ln = np.log(y_col)
X_exp = np.c_[np.ones(n), X_col]
theta_exp = normal_equation(X_exp, y_ln)
A_exp = np.exp(theta_exp[0][0])
B_exp = theta_exp[1][0]
y_pred_exp = A_exp * np.exp(B_exp * X_col)

# Bước 4: Tính toán sai số (MSE và RMSE)
def calculate_errors(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse, np.sqrt(mse)

mse_lin,  rmse_lin  = calculate_errors(y_col, y_pred_linear)
mse_quad, rmse_quad = calculate_errors(y_col, y_pred_quad)
mse_exp,  rmse_exp  = calculate_errors(y_col, y_pred_exp)

# Bước 5: In bảng so sánh
models     = ["Tuyến tính (Bậc 1)", "Đa thức (Bậc 2)", "Hàm mũ"]
mse_list   = [mse_lin,  mse_quad,  mse_exp]
rmse_list  = [rmse_lin, rmse_quad, rmse_exp]

print("\nBẢNG SO SÁNH SAI SỐ GIỮA CÁC MÔ HÌNH:")
print("-" * 62)
print(f"{'Mô hình':<22} {'MSE':>20} {'RMSE':>15}")
print("-" * 62)
for name, mse, rmse in zip(models, mse_list, rmse_list):
    print(f"{name:<22} {mse:>20,.2f} {rmse:>15,.2f}")
print("-" * 62)

best_idx    = int(np.argmin(rmse_list))
best_name   = models[best_idx]
best_rmse   = rmse_list[best_idx]
second_rmse = sorted(rmse_list)[1]

print(f"\nKẾT LUẬN: Mô hình RMSE nhỏ nhất là [{best_name}]")
print(f"  • Tuyến tính    : RMSE = {rmse_lin:>12,.2f} USD")
print(f"  • Đa thức Bậc 2 : RMSE = {rmse_quad:>12,.2f} USD")
print(f"  • Hàm mũ        : RMSE = {rmse_exp:>12,.2f} USD")

diff_pct = (second_rmse - best_rmse) / best_rmse * 100
if diff_pct < 1.0:
    simplest = models[0]  # Tuyến tính luôn là mô hình đơn giản nhất
    print(f"\n⚠️  Cải thiện so với mô hình kế tiếp chỉ {diff_pct:.4f}% → không đáng kể")
    print(f"✔  Khuyến nghị: Nên dùng [{simplest}] vì đơn giản hơn, kết quả tương đương")
else:
    print(f"\n✔  Cải thiện rõ ràng {diff_pct:.2f}% → Nên dùng [{best_name}]")
print()

# Bước 6: Biểu đồ trực quan
COLORS = {"lin": "#2563EB", "quad": "#16A34A", "exp": "#DC2626"}

a0_lin, a1_lin         = theta_linear.flatten()
a0_q,   a1_q,   a2_q  = theta_quad.flatten()

x_line      = np.linspace(X.min(), X.max(), 400)
y_line_lin  = a0_lin + a1_lin * x_line
y_line_quad = a0_q   + a1_q   * x_line + a2_q * x_line**2
y_line_exp  = A_exp  * np.exp(B_exp * x_line)

fig = plt.figure(figsize=(14, 9), facecolor="#F8FAFC")
fig.suptitle("So sánh mô hình hồi quy\nDiện tích nhà (sq ft) – Giá bán (USD)",
             fontsize=14, fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.88, bottom=0.08, left=0.07, right=0.97)

fmt = plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K")

subplots_cfg = [
    (gs[0,0], "Tuyến tính (Bậc 1)", y_line_lin,  COLORS["lin"],
     f"y={a0_lin:,.0f}+{a1_lin:.2f}x", rmse_lin),
    (gs[0,1], "Đa thức (Bậc 2)",    y_line_quad, COLORS["quad"],
     f"y={a0_q:,.0f}+{a1_q:.2f}x+{a2_q:.5f}x²", rmse_quad),
    (gs[0,2], "Hàm mũ",             y_line_exp,  COLORS["exp"],
     f"y={A_exp:.0f}·e^({B_exp:.5f}x)", rmse_exp),
]

for spec, title, yl, color, eq_lbl, rmse_v in subplots_cfg:
    ax = fig.add_subplot(spec)
    ax.scatter(X, y, color="#64748B", s=40, zorder=5, label="Dữ liệu thực tế")
    ax.plot(x_line, yl, color=color, lw=2.2,
            label=f"{eq_lbl}\nRMSE={rmse_v:,.0f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Diện tích (sq ft)")
    ax.set_ylabel("Giá bán (USD)")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

# Subplot so sánh tất cả mô hình
ax_all = fig.add_subplot(gs[1, 0:2])
ax_all.scatter(X, y, color="#64748B", s=50, zorder=5, label="Dữ liệu thực tế")
ax_all.plot(x_line, y_line_lin,  color=COLORS["lin"],  lw=2,   ls="--",
            label=f"Tuyến tính (RMSE={rmse_lin:,.0f})")
ax_all.plot(x_line, y_line_quad, color=COLORS["quad"], lw=2.5, ls="-",
            label=f"Bậc 2 (RMSE={rmse_quad:,.0f}) ✔ tốt nhất")
ax_all.plot(x_line, y_line_exp,  color=COLORS["exp"],  lw=2,   ls="-.",
            label=f"Hàm mũ (RMSE={rmse_exp:,.0f})")
ax_all.set_title("So sánh tất cả mô hình", fontweight="bold")
ax_all.set_xlabel("Diện tích (sq ft)")
ax_all.set_ylabel("Giá bán (USD)")
ax_all.yaxis.set_major_formatter(fmt)
ax_all.legend(fontsize=9)
ax_all.grid(alpha=0.3)

# Subplot bar chart MSE & RMSE
ax_bar = fig.add_subplot(gs[1, 2])
xpos = np.arange(3)
w = 0.35
bar_colors = [COLORS["lin"], COLORS["quad"], COLORS["exp"]]

b1b = ax_bar.bar(xpos - w/2, [v/1e9 for v in mse_list],  w,
                 label="MSE (×10⁹)", color=bar_colors, alpha=0.5)
b2b = ax_bar.bar(xpos + w/2, [v/1e3 for v in rmse_list], w,
                 label="RMSE (×10³)", color=bar_colors, alpha=0.9, hatch="//")

for bar in list(b1b) + list(b2b):
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

ax_bar.set_xticks(xpos)
ax_bar.set_xticklabels(["Tuyến tính", "Bậc 2", "Hàm mũ"], fontsize=8.5)
ax_bar.set_title("So sánh MSE & RMSE", fontweight="bold")
ax_bar.legend(fontsize=8)
ax_bar.grid(alpha=0.3)

plt.savefig("bieu_do_hoi_quy.png", dpi=150, bbox_inches="tight")
plt.show()
