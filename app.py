import streamlit as st
import numpy as np
from pdf2image import convert_from_bytes
from PIL import ImageDraw, ImageFont, Image
import io, cv2

def detect_staff_lines(pil_img):
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 1. 二値化（文字や線を白=255、背景を黒=0に反転）
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. 水平投影：各行（横方向）の白ピクセル(255)の合計を計算
    # これにより、Y座標ごとの「線の密集度」のグラフが作られます
    horizontal_sum = np.sum(thresh, axis=1)

    # 3. 閾値（しきいち）の設定：画像の横幅の「何％」が直線なら五線とみなすか
    width = thresh.shape[1]
    # 例：横幅の50%以上が黒（255）で埋まっている行を探す
    threshold_value = width * 255 * 0.5 

    # 4. 閾値を超えたY座標（行）をすべて取得
    y_coordinates = np.where(horizontal_sum > threshold_value)[0]

    # -- 画面確認用（元の画像に赤線を引く） --
    result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 見つけたY座標に、端から端まで赤い横線を引く
    for y in y_coordinates:
        cv2.line(result_img, (0, y), (width, y), (255, 0, 0), 1)

    return Image.fromarray(result_img)

# ページの設定
st.set_page_config(page_title="ドレミ自動付与ツール", layout="centered")

st.title("🎼 楽譜ドレミ自動付与ツール")
st.write("PDFの楽譜をアップロードすると、自動で音階を解析します。")

# 1. ファイルアップロード機能
uploaded_file = st.file_uploader("PDF形式の楽譜を選択してください", type=["pdf"])

if uploaded_file is not None:
    st.success("ファイルを受け取りました！解析を開始します...")

    # 2. PDFを表示用に画像変換 (プレビュー)
    # ※実際の運用には poppler のインストールが必要です
    images = convert_from_bytes(uploaded_file.read())
    
    for i, image in enumerate(images):
        st.image(image, caption=f"{i+1}ページ目", use_column_width=True)

# 3. 画像への書き込み処理
    with st.spinner('五線を検出中...'):
        processed_images = []
        
        for i, img in enumerate(images):
            # さっき作ったOpenCVの関数で五線を検出！
            result_img = detect_staff_lines(img)
            
            processed_images.append(result_img)
            
            # 結果を画面にプレビュー表示
            st.image(result_img, caption=f"{i+1}ページ目 (五線検出後)", use_column_width=True)

    st.success("処理が完了しました！")

    # 4. 処理結果をPDFとしてまとめる
    pdf_byte_arr = io.BytesIO()
    # 複数ページある場合は save_all=True で結合します
    processed_images[0].save(
        pdf_byte_arr, format='PDF', 
        save_all=True, append_images=processed_images[1:]
    )
    
    # ダウンロードボタン
    st.download_button(
        label="ドレミ付き楽譜をダウンロード",
        data=pdf_byte_arr.getvalue(),
        file_name="doremi_score.pdf",
        mime="application/pdf"
    )
