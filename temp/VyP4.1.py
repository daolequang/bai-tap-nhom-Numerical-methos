import streamlit as st
import pandas as pd

st.set_page_config(page_title="Nhập dữ liệu - Ý 1", layout="centered")

st.header("Nhập dữ liệu đầu vào")

# 1. Tạo lựa chọn phương thức nhập
input_method = st.radio(
    "Chọn cách thức cung cấp dữ liệu:",
    ("Tải lên file CSV", "Nhập thủ công (x, y)")
)

df = None # Biến lưu trữ bảng dữ liệu

# --- TRƯỜNG HỢP 1: TẢI FILE CSV ---
if input_method == "Tải lên file CSV":
    uploaded_file = st.file_uploader("Chọn tệp CSV (Lưu ý: Cột 1 là x, cột 2 là y)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Đã tải dữ liệu từ file thành công!")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

# --- TRƯỜNG HỢP 2: NHẬP THỦ CÔNG ---
else:
    st.info("Nhập mỗi cặp x, y trên một dòng, cách nhau bởi dấu phẩy. Ví dụ: 1, 2.5")
    raw_text = st.text_area("Vùng nhập dữ liệu:", placeholder="1, 10\n2, 20\n3, 35", height=150)
    
    if raw_text:
        try:
            # Tách dòng và tách dấu phẩy, chuyển về dạng số
            lines = [line.split(',') for line in raw_text.strip().split('\n') if line.strip()]
            df = pd.DataFrame(lines, columns=['x', 'y']).astype(float)
            st.success("Đã ghi nhận dữ liệu nhập tay!")
        except ValueError:
            st.error("Lỗi định dạng: Hãy đảm bảo bạn chỉ nhập số và dùng dấu phẩy để ngăn cách.")

# --- HIỂN THỊ KẾT QUẢ ---
st.divider()
if df is not None:
    st.subheader("Bảng dữ liệu đã ghi nhận:")
    st.dataframe(df, use_container_width=True)
    st.info(f"Tổng cộng có **{len(df)}** cặp điểm dữ liệu.")
else:
    st.warning("Chưa có dữ liệu. Vui lòng nhập hoặc tải file để tiếp tục.")