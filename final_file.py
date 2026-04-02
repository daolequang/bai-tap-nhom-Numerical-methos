import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_csv_data(file):
    """
    Đọc CSV linh hoạt:
    - có / không có header
    - dấu phân cách , hoặc ;
    - tự lấy 2 cột đầu
    - tự ép kiểu số
    """
    try:
        # Đọc linh hoạt dấu phân cách
        df = pd.read_csv(file, sep=None, engine='python', header=None)

        # Bỏ dòng rỗng
        df = df.dropna()

        # Kiểm tra số cột
        if df.shape[1] < 2:
            raise ValueError("File CSV phải có ít nhất 2 cột.")

        # Chỉ lấy 2 cột đầu
        df = df.iloc[:, :2].copy()
        df.columns = ['x', 'y']

        # Ép kiểu số
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')

        # Loại dòng không hợp lệ
        df = df.dropna()

        if len(df) == 0:
            raise ValueError("Không có dữ liệu hợp lệ sau khi xử lý.")

        return df

    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file CSV: {e}")

#cấu hình trang
st.set_page_config(page_title="Hệ thống Xấp xỉ Hàm số", page_icon="📈", layout="wide")


#toán
def gauss_elimination(matrix):
    m = len(matrix)
    for col in range(m):
        max_row = col
        for row in range(col + 1, m):
            if abs(matrix[row][col]) > abs(matrix[max_row][col]):
                max_row = row
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]

        pivot = matrix[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Ma trận suy biến - Không thể giải hệ phương trình với dữ liệu này.")

        for row in range(col + 1, m):
            factor = matrix[row][col] / pivot
            for j in range(col, m + 1):
                matrix[row][j] -= factor * matrix[col][j]

    solution = [0.0] * m
    for row in range(m - 1, -1, -1):
        solution[row] = matrix[row][m]
        for j in range(row + 1, m):
            solution[row] -= matrix[row][j] * solution[j]
        solution[row] /= matrix[row][row]

    return solution

def calculate_errors(y_true, y_pred):
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)
    return mse, math.sqrt(mse)

def linear_regression(x, y, n):
    sum_x  = sum(x)
    sum_y  = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))
    matrix = [[float(n), sum_x, sum_y], [sum_x, sum_x2, sum_xy]]
    return gauss_elimination(matrix) #trả về a0, a1 (y = a0 + a1*x)

def quadratic_regression(x, y, n):
    sum_x, sum_y = sum(x), sum(y)
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
    return gauss_elimination(matrix) #trả về a0, a1, a2 (y = a0 + a1*x + a2*x²)

def exponential_regression(x, y, n):
    Y = [math.log(yi) for yi in y]
    sum_x, sum_Y = sum(x), sum(Y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xY = sum(x[i]*Y[i] for i in range(n))
    matrix = [[float(n), sum_x, sum_Y], [sum_x, sum_x2, sum_xY]]
    A, b = gauss_elimination(matrix)
    a = math.exp(A)
    return a, b #trả về a, b (y = a * e^(b*x))

#giao diện
st.title("Hệ thống Phân tích & Xấp xỉ Hàm số")
st.markdown("Hệ thống giải hệ phương trình chuẩn bằng phương pháp khử Gauss (tự xây dựng).")
st.markdown("---")

#tạo 2 tab
tab1, tab2 = st.tabs(["CÔNG CỤ TÙY CHỈNH (Nhập liệu)", "BÀI TOÁN ĐÁNH GIÁ (So sánh 3 mô hình)"])

#nhập liệu cũ
with tab1:
    st.header("1. Cài đặt & Nhập liệu")
    phuong_thuc_nhap = st.selectbox(
        "Chọn phương thức nhập:",
        ("Tải lên file CSV", "Nhập thủ công (x, y)"),
        key="nhap_lieu_tab1"
    )

    bang_du_lieu = None 
    
    if phuong_thuc_nhap == "Tải lên file CSV":
        file_tai_len = st.file_uploader("Chọn tệp CSV (Cột 1: x, Cột 2: y)", type=["csv"], key="file_t1")
        if file_tai_len is not None:
            try:
                try:
                    bang_du_lieu = load_csv_data(file_tai_len)
                    st.success(f"Tải dữ liệu thành công! ({len(bang_du_lieu)} điểm)")
                except Exception as loi:
                    st.error(str(loi))
            except Exception as loi:
                st.error(f"Lỗi khi đọc file: {loi}")

    else:
        st.info("Nhập mỗi cặp x, y trên một dòng, cách nhau bởi dấu phẩy.")
        du_lieu_nhap_tay = st.text_area("Dữ liệu (x, y):", placeholder="1, 10\n2, 20\n3, 35", height=150, key="text_t1")
        
        if du_lieu_nhap_tay:
            try:
                cac_dong = [dong.split(',') for dong in du_lieu_nhap_tay.strip().split('\n') if dong.strip()]
                cac_dong_hop_le = [dong for dong in cac_dong if len(dong) == 2]
                
                if len(cac_dong_hop_le) > 0:
                    bang_du_lieu = pd.DataFrame(cac_dong_hop_le, columns=['x', 'y']).astype(float)
                    st.success(f"Đã ghi nhận {len(bang_du_lieu)} điểm dữ liệu!")
                else:
                    st.warning("Chưa có dữ liệu hợp lệ.")
            except ValueError:
                st.error("Lỗi định dạng! Chỉ dùng số và dấu phẩy.")

    st.markdown("---")

    if bang_du_lieu is not None and len(bang_du_lieu) > 1:
        du_lieu_x = bang_du_lieu['x'].values
        du_lieu_y = bang_du_lieu['y'].values
        n_points = len(du_lieu_x)

        cot_du_lieu, cot_mo_hinh = st.columns([1, 2])
        
        with cot_du_lieu:
            st.subheader("Dữ liệu đầu vào")
            st.dataframe(bang_du_lieu, use_container_width=True, height=180)

        with cot_mo_hinh:
            st.subheader("Cấu hình Mô hình")
            loai_ham = st.selectbox(
                "2. Chọn dạng hàm xấp xỉ:",
                ("Tuyến tính (y = a0 + a1*x)", "Đa thức bậc 2 (y = a0 + a1*x + a2*x²)", "Hàm mũ (y = a * e^(bx))")
            )
            
            chuoi_phuong_trinh = ""
            ham_du_bao = None
            mau_do_thi = '#FF4B4B'

            try:
                #khử gauss
                if "Tuyến tính" in loai_ham:
                    a0, a1 = linear_regression(du_lieu_x, du_lieu_y, n_points)
                    chuoi_phuong_trinh = f"y = {a1:.4f}x {'+' if a0 >=0 else '-'} {abs(a0):.4f}"
                    ham_du_bao = lambda x: a0 + a1 * x
                    mau_do_thi = '#FF4B4B'

                elif "bậc 2" in loai_ham:
                    if n_points < 3:
                        st.error("Cần ít nhất 3 điểm dữ liệu để giải hệ phương trình bậc 2!")
                    else:
                        a0, a1, a2 = quadratic_regression(du_lieu_x, du_lieu_y, n_points)
                        chuoi_phuong_trinh = f"y = {a2:.4f}x² {'+' if a1 >=0 else '-'} {abs(a1):.4f}x {'+' if a0 >=0 else '-'} {abs(a0):.4f}"
                        ham_du_bao = lambda x: a0 + a1*(x) + a2*(x**2)
                        mau_do_thi = '#0068C9' 

                elif "Hàm mũ" in loai_ham:
                    if any(du_lieu_y <= 0):
                        st.error("Dữ liệu Y phải lớn hơn 0 để xấp xỉ hàm mũ (do cần tính log)!")
                    else:
                        a, b = exponential_regression(du_lieu_x, du_lieu_y, n_points)
                        chuoi_phuong_trinh = f"y = {a:.4f} * e^({b:.4f}x)"
                        ham_du_bao = lambda x: a * np.exp(b * x)
                        mau_do_thi = '#29B09D'

                if ham_du_bao:
                    st.info(f"**Phương trình tối ưu (Tính bằng Khử Gauss):**\n### {chuoi_phuong_trinh}")
            except Exception as loi:
                st.error(f"Lỗi tính toán: {loi}")

        st.markdown("---")

        if ham_du_bao is not None:
            st.subheader("Trực quan hóa Đồ thị")
            hinh_ve, truc = plt.subplots(figsize=(10, 4))
            truc.set_facecolor('#F8F9FB')
            
            truc.scatter(du_lieu_x, du_lieu_y, color='#31333F', label='Dữ liệu thực tế', zorder=5, s=60, alpha=0.7)
            
            x_nho_nhat, x_lon_nhat = np.min(du_lieu_x), np.max(du_lieu_x)
            khoang_dem = (x_lon_nhat - x_nho_nhat) * 0.1 if x_lon_nhat != x_nho_nhat else 1
            x_muot = np.linspace(x_nho_nhat - khoang_dem, x_lon_nhat + khoang_dem, 200)
            y_muot = [ham_du_bao(xi) for xi in x_muot]
            
            truc.plot(x_muot, y_muot, color=mau_do_thi, label='Đường xấp xỉ', linewidth=2.5)
            
            truc.set_xlabel('Trục X', fontweight='bold')
            truc.set_ylabel('Trục Y', fontweight='bold')
            truc.legend()
            truc.grid(True, linestyle='--', alpha=0.5, color='white')
            
            st.pyplot(hinh_ve)
            st.markdown("---")
            
            st.subheader("Dự báo Giá trị (Prediction)")
            cot_nhap, cot_ket_qua = st.columns([1, 1])
            with cot_nhap:
                x_moi = st.number_input("3. Nhập giá trị $x_{new}$:", value=float(x_nho_nhat), step=0.1)
            with cot_ket_qua:
                y_moi = ham_du_bao(x_moi)
                st.metric(label=f"Giá trị dự báo $y_{{new}}$ khi $x = {x_moi}$", value=f"{y_moi:.4f}")

    elif bang_du_lieu is None or len(bang_du_lieu) == 0:
        st.info("Hãy cung cấp dữ liệu đầu vào ở phía trên để bắt đầu phân tích.")
    elif len(bang_du_lieu) <= 1:
        st.warning("Vui lòng nhập ít nhất 2 điểm dữ liệu để có thể xấp xỉ hàm số.")


#code 2
with tab2:
    st.header("Giải Bài Toán Hồi Quy Thực Nghiệm & So Sánh")
    st.markdown("""
    - **Thuật toán:** Giải hệ phương trình chuẩn bằng phương pháp khử Gauss (tự xây dựng).
    - **Đánh giá:** Tính toán và so sánh theo MSE, RMSE giữa 3 mô hình (Tuyến tính, Bậc 2, Hàm mũ).
    """)

    phuong_thuc_nhap_tab2 = st.selectbox(
        "Chọn nguồn dữ liệu để đánh giá:",
        ("Dữ liệu mẫu (Case Study Giá nhà - 30 điểm)", "Tải lên file CSV", "Nhập thủ công (x, y)"),
        key="nhap_lieu_tab2"
    )

    x_data_f2 = []
    y_data_f2 = []
    du_lieu_hop_le_tab2 = False

    if phuong_thuc_nhap_tab2 == "Dữ liệu mẫu (Case Study Giá nhà - 30 điểm)":
        x_data_f2 = [2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494,
                     1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031,
                     1767, 1888, 1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637]
        y_data_f2 = [399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999,
                     212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900,
                     299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900,
                     573900, 249900, 464500, 469000, 475000, 299900]
        du_lieu_hop_le_tab2 = True

    elif phuong_thuc_nhap_tab2 == "Tải lên file CSV":
        file_tai_len_t2 = st.file_uploader("Chọn tệp CSV (Cột 1: x, Cột 2: y)", type=["csv"], key="file_t2")
        if file_tai_len_t2 is not None:
            try:
                try:
                    bang_tam_t2 = load_csv_data(file_tai_len_t2)
                    x_data_f2 = bang_tam_t2['x'].tolist()
                    y_data_f2 = bang_tam_t2['y'].tolist()
                    du_lieu_hop_le_tab2 = True
                    st.success(f"Tải dữ liệu thành công! ({len(bang_tam_t2)} điểm)")
                except Exception as loi:
                    st.error(str(loi))
            except Exception as loi:
                st.error(f"Lỗi khi đọc file: {loi}")

    else:
        st.info("Nhập mỗi cặp x, y trên một dòng, cách nhau bởi dấu phẩy.")
        du_lieu_nhap_tay_t2 = st.text_area("Dữ liệu (x, y):", placeholder="2104, 399900\n1600, 329900\n2400, 369000", height=150, key="text_t2")
        if du_lieu_nhap_tay_t2:
            try:
                cac_dong = [dong.split(',') for dong in du_lieu_nhap_tay_t2.strip().split('\n') if dong.strip()]
                cac_dong_hop_le = [dong for dong in cac_dong if len(dong) == 2]
                if len(cac_dong_hop_le) > 0:
                    x_data_f2 = [float(d[0]) for d in cac_dong_hop_le]
                    y_data_f2 = [float(d[1]) for d in cac_dong_hop_le]
                    du_lieu_hop_le_tab2 = True
                    st.success(f"Đã ghi nhận {len(x_data_f2)} điểm dữ liệu!")
                else:
                    st.warning("Chưa có dữ liệu hợp lệ.")
            except ValueError:
                st.error("Lỗi định dạng! Chỉ dùng số và dấu phẩy.")

    #tiến hành tính toán nếu dữ liệu hợp lệ
    if du_lieu_hop_le_tab2:
        if len(x_data_f2) < 3:
            st.warning("Vui lòng cung cấp ít nhất 3 điểm dữ liệu để có thể so sánh cả 3 mô hình (Bậc 2 cần tối thiểu 3 điểm).")
        elif any(y <= 0 for y in y_data_f2):
            st.error("Dữ liệu Y chứa giá trị ≤ 0. Hàm mũ không thể tính toán (do cần lấy logarit). Vui lòng sử dụng tập dữ liệu có giá trị Y > 0 để so sánh.")
        else:
            n_f2 = len(x_data_f2)

            #tính toán các mô hình
            a0_lin, a1_lin = linear_regression(x_data_f2, y_data_f2, n_f2)
            y_pred_linear = [a0_lin + a1_lin * x for x in x_data_f2]
            mse_lin, rmse_lin = calculate_errors(y_data_f2, y_pred_linear)

            a0_q, a1_q, a2_q = quadratic_regression(x_data_f2, y_data_f2, n_f2)
            y_pred_quad = [a0_q + a1_q * x + a2_q * x**2 for x in x_data_f2]
            mse_quad, rmse_quad = calculate_errors(y_data_f2, y_pred_quad)

            a_exp, b_exp = exponential_regression(x_data_f2, y_data_f2, n_f2)
            y_pred_exp = [a_exp * math.exp(b_exp * x) for x in x_data_f2]
            mse_exp, rmse_exp = calculate_errors(y_data_f2, y_pred_exp)

            #tự động thay đổi đơn vị theo dạng dữ liệu
            is_sample_data = (phuong_thuc_nhap_tab2 == "Dữ liệu mẫu (Case Study Giá nhà - 30 điểm)")
            don_vi_y = " (USD)" if is_sample_data else ""
            nhan_truc_x = "Diện tích (sq ft)" if is_sample_data else "Trục X"
            nhan_truc_y = "Giá bán (USD)" if is_sample_data else "Trục Y"

            #đánh giá và hiển thị bảng so sánh
            st.subheader("1. Bảng So Sánh Sai Số Các Mô Hình")
            df_eval = pd.DataFrame({
                "Mô hình": ["Tuyến tính (y = a0 + a1*x)", "Đa thức Bậc 2", "Hàm mũ (y = a*e^(bx))"],
                "MSE": [mse_lin, mse_quad, mse_exp],
                f"RMSE{don_vi_y}": [rmse_lin, rmse_quad, rmse_exp]
            })
            st.dataframe(df_eval.style.format({"MSE": "{:,.2f}", f"RMSE{don_vi_y}": "{:,.2f}"}), use_container_width=True)

            models_eval = [("Tuyến tính", rmse_lin), ("Đa thức Bậc 2", rmse_quad), ("Hàm mũ", rmse_exp)]
            models_eval.sort(key=lambda t: t[1])
            best_model, best_rmse = models_eval[0]
            
            st.success(f"**KẾT LUẬN:** Mô hình biểu diễn dữ liệu tốt nhất là **[{best_model}]** với RMSE nhỏ nhất: **{best_rmse:,.2f}{don_vi_y}**")

            #phân tích trực quan
            st.subheader("2. Phân tích Trực quan")
            
            X_np = np.array(x_data_f2)
            y_np = np.array(y_data_f2)
            x_line = np.linspace(X_np.min(), X_np.max(), 400)

            y_line_lin  = a0_lin + a1_lin * x_line
            y_line_quad = a0_q   + a1_q * x_line + a2_q * x_line**2
            y_line_exp  = a_exp  * np.exp(b_exp * x_line)

            COLORS = {"lin": "#2563EB", "quad": "#16A34A", "exp": "#DC2626"}

            fig = plt.figure(figsize=(14, 9), facecolor="#F8FAFC")
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35, top=0.88, bottom=0.08, left=0.07, right=0.97)
            
            if np.max(y_np) >= 10000:
                fmt = plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K")
            else:
                fmt = plt.FuncFormatter(lambda v, _: f"{v:,.1f}")

            subplots_cfg = [
                (gs[0,0], "Tuyến tính (Bậc 1)", y_line_lin, COLORS["lin"], f"y={a0_lin:,.0f}+{a1_lin:.2f}x", rmse_lin),
                (gs[0,1], "Đa thức (Bậc 2)", y_line_quad, COLORS["quad"], f"y={a0_q:,.0f}+{a1_q:.2f}x...", rmse_quad),
                (gs[0,2], "Hàm mũ", y_line_exp, COLORS["exp"], f"y={a_exp:.0f}·e^({b_exp:.5f}x)", rmse_exp),
            ]

            for spec, title, yl, color, eq_lbl, rmse_v in subplots_cfg:
                ax = fig.add_subplot(spec)
                ax.scatter(X_np, y_np, color="#64748B", s=40, zorder=5, label="Dữ liệu")
                ax.plot(x_line, yl, color=color, lw=2.2, label=f"{eq_lbl}\nRMSE={rmse_v:,.1f}")
                ax.set_title(title, fontweight="bold")
                ax.set_xlabel(nhan_truc_x)
                ax.set_ylabel(nhan_truc_y)
                ax.yaxis.set_major_formatter(fmt)
                ax.legend(fontsize=8, loc="upper left")
                ax.grid(alpha=0.3)

            ax_all = fig.add_subplot(gs[1, 0:2])
            ax_all.scatter(X_np, y_np, color="#64748B", s=50, zorder=5, label="Dữ liệu thực tế")
            ax_all.plot(x_line, y_line_lin, color=COLORS["lin"], lw=2, ls="--", label=f"Tuyến tính (RMSE={rmse_lin:,.1f})")
            ax_all.plot(x_line, y_line_quad, color=COLORS["quad"], lw=2.5, ls="-", label=f"Bậc 2 (RMSE={rmse_quad:,.1f})")
            ax_all.plot(x_line, y_line_exp, color=COLORS["exp"], lw=2, ls="-.", label=f"Hàm mũ (RMSE={rmse_exp:,.1f})")
            ax_all.set_title("So sánh trực quan các mô hình", fontweight="bold")
            ax_all.set_xlabel(nhan_truc_x)
            ax_all.set_ylabel(nhan_truc_y)
            ax_all.yaxis.set_major_formatter(fmt)
            ax_all.legend(fontsize=9)
            ax_all.grid(alpha=0.3)

            ax_bar = fig.add_subplot(gs[1, 2])
            xpos = np.arange(3)
            w = 0.4
            rmse_list = [rmse_lin, rmse_quad, rmse_exp]
            bar_colors = [COLORS["lin"], COLORS["quad"], COLORS["exp"]]

            #xử lý tự động nhãn dán Bar chart
            if np.max(rmse_list) >= 10000:
                bar_vals = [v/1e3 for v in rmse_list]
                bar_label = f"RMSE (×10³{don_vi_y})"
                suffix = "K"
            else:
                bar_vals = rmse_list
                bar_label = f"RMSE{don_vi_y}"
                suffix = ""

            b2b = ax_bar.bar(xpos, bar_vals, w, label=bar_label, color=bar_colors, alpha=0.9, hatch="//")
            for bar in b2b:
                h = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h * 1.02, f"{h:,.1f}{suffix}", ha="center", va="bottom", fontsize=8)

            ax_bar.set_xticks(xpos)
            ax_bar.set_xticklabels(["Tuyến tính", "Bậc 2", "Hàm mũ"], fontsize=9)
            ax_bar.set_title("So sánh RMSE", fontweight="bold")
            ax_bar.grid(axis='y', alpha=0.3)

            st.pyplot(fig)