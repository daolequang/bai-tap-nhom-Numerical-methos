#cách dùng mở terminal ghi, ko ghi []
#streamlit run [đường dẫn file code]
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#cấu hình trang
st.set_page_config(page_title="Xấp xỉ hàm số", page_icon="📈", layout="wide")

#tiêu đề 
st.title("📈 Bảng điều khiển Xấp xỉ Hàm số")
st.markdown("Ứng dụng hỗ trợ tìm phương trình thực nghiệm từ dữ liệu rời rạc và trực quan hóa kết quả.")
st.markdown("---")

#nhập liệu
st.header("1. Cài đặt & Nhập liệu")
phuong_thuc_nhap = st.selectbox(
    "Chọn phương thức nhập:",
    ("Tải lên file CSV", "Nhập thủ công (x, y)")
)

bang_du_lieu = None  #biến lưu dữ liệu
#phần của vỹ nhưng tui việt hóa lại cho dễ hiểu =))
if phuong_thuc_nhap == "Tải lên file CSV":
    file_tai_len = st.file_uploader("Chọn tệp CSV (Cột 1: x, Cột 2: y)", type=["csv"])
    if file_tai_len is not None:
        try:
            bang_tam = pd.read_csv(file_tai_len, header=None)
            if len(bang_tam.columns) >= 2:
                bang_du_lieu = pd.DataFrame({'x': bang_tam.iloc[:, 0], 'y': bang_tam.iloc[:, 1]}).astype(float)
                st.success("Tải dữ liệu thành công!")
            else:
                st.error("File CSV phải có ít nhất 2 cột.")
        except Exception as loi:
            st.error(f"Lỗi khi đọc file: {loi}")

else:
    st.info("Nhập mỗi cặp x, y trên một dòng, cách nhau bởi dấu phẩy.")
    du_lieu_nhap_tay = st.text_area("Dữ liệu (x, y):", placeholder="1, 10\n2, 20\n3, 35", height=150)
    
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

#chia thành 2 cột
    cot_du_lieu, cot_mo_hinh = st.columns([1, 2])
    
    with cot_du_lieu:
        st.subheader("Dữ liệu đầu vào")
        st.dataframe(bang_du_lieu, use_container_width=True, height=180)

    with cot_mo_hinh:
        st.subheader("Cấu hình Mô hình")
        loai_ham = st.selectbox(
            "2. Chọn dạng hàm xấp xỉ:",
            ("Tuyến tính (y = ax + b)", "Đa thức bậc 2 (y = ax² + bx + c)", "Hàm mũ (y = a * e^(bx))")
        )
        
        chuoi_phuong_trinh = ""
        ham_du_bao = None
        mau_do_thi = '#FF4B4B'

        #toán học
        try:
            if "Tuyến tính" in loai_ham:
                he_so = np.polyfit(du_lieu_x, du_lieu_y, 1)
                a, b = he_so
                chuoi_phuong_trinh = f"y = {a:.4f}x {'+' if b >=0 else '-'} {abs(b):.4f}"
                ham_du_bao = lambda x: a * x + b
                mau_do_thi = '#FF4B4B'

            elif "bậc 2" in loai_ham:
                he_so = np.polyfit(du_lieu_x, du_lieu_y, 2)
                a, b, c = he_so
                chuoi_phuong_trinh = f"y = {a:.4f}x² {'+' if b >=0 else '-'} {abs(b):.4f}x {'+' if c >=0 else '-'} {abs(c):.4f}"
                ham_du_bao = lambda x: a * (x**2) + b * x + c
                mau_do_thi = '#0068C9' 

            elif "Hàm mũ" in loai_ham:
                if any(du_lieu_y <= 0):
                    st.error("Dữ liệu Y phải lớn hơn 0 để xấp xỉ hàm mũ!")
                else:
                    def ham_mu(x, a, b):
                        return a * np.exp(b * x)
                    tham_so_toi_uu, _ = curve_fit(ham_mu, du_lieu_x, du_lieu_y)
                    a, b = tham_so_toi_uu
                    chuoi_phuong_trinh = f"y = {a:.4f} * e^({b:.4f}x)"
                    ham_du_bao = lambda x: a * np.exp(b * x)
                    mau_do_thi = '#29B09D'

            if ham_du_bao:
                st.info(f"**Phương trình tối ưu:**\n### {chuoi_phuong_trinh}")
        except Exception as loi:
            st.error(f"Lỗi tính toán: {loi}")

    st.markdown("---")

    #check xem có thành công hay ko 
    if ham_du_bao is not None:
        st.subheader("Trực quan hóa Đồ thị")
        
        #tạo matplot
        hinh_ve, truc = plt.subplots(figsize=(10, 4))
        truc.set_facecolor('#F8F9FB')
        
        #dữ liệu gốc
        truc.scatter(du_lieu_x, du_lieu_y, color='#31333F', label='Dữ liệu thực tế', zorder=5, s=60, alpha=0.7)
        
        #đường xấp xỉ
        x_nho_nhat, x_lon_nhat = np.min(du_lieu_x), np.max(du_lieu_x)
        khoang_dem = (x_lon_nhat - x_nho_nhat) * 0.1 if x_lon_nhat != x_nho_nhat else 1
        x_muot = np.linspace(x_nho_nhat - khoang_dem, x_lon_nhat + khoang_dem, 200)
        y_muot = ham_du_bao(x_muot)
        
        truc.plot(x_muot, y_muot, color=mau_do_thi, label='Đường xấp xỉ', linewidth=2.5)
        
        #màu mè trang trí
        truc.set_xlabel('Trục X', fontweight='bold')
        truc.set_ylabel('Trục Y', fontweight='bold')
        truc.legend()
        truc.grid(True, linestyle='--', alpha=0.5, color='white')
        
        st.pyplot(hinh_ve)

        st.markdown("---")
        
        #dự báo giá trị
        st.subheader("Dự báo Giá trị")
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
