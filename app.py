import streamlit as st
import numpy as np
from pdf2image import convert_from_bytes
from PIL import ImageDraw, ImageFont, Image
import io, cv2

def detect_staff_lines(pil_img):
    # 1. PIL画像(Pillow)をOpenCVで扱える形式(NumPy配列)に変換
    img_array = np.array(pil_img)
    
    # 2. グレースケール（白黒）に変換
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 3. 二値化（白黒をはっきりさせ、色を反転する。線＝白、背景＝黒にするため）
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 4. 横線を抽出するための「横長のフィルター」を作成
    # 画像の幅の1/40くらいの長さを基準にします
    width = thresh.shape[1]
    kernel_len = width // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    # 5. モルフォロジー変換（横線だけを残す画像処理の魔法）
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # -- ここからは画面確認用（元の画像に赤線を引く） --
    # 線の塊（輪郭）を見つける
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 結果を描き込む用の画像（カラー）を用意
    result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
# 見つけた線の上に赤い線を引く
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 輪郭の面積（ピクセル数）を取得
        area = cv2.contourArea(contour)
        
        # 横幅が十分に長く、かつ「平均の太さ (面積 ÷ 横幅)」が細いものだけを残す
        # ※ 5 の部分は線の太さに合わせて微調整してください（大抵は 2〜5 くらいで収まります）
        if w > width // 4 and (area / w) < 1: 
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2) # 赤枠を描画

    # OpenCVの画像をPIL画像に戻して返す
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
