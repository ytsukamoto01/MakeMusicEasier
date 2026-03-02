import streamlit as st
from pdf2image import convert_from_bytes
from PIL import ImageDraw, ImageFont
import io

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

# 3. 画像への書き込み処理 (Pillowを使用)
    with st.spinner('ドレミを書き込み中...'):
        processed_images = []
        
        # フォントの設定 (Streamlit Cloudのデフォルトフォントを使用)
        # ※日本語フォントがない場合、文字化け（豆腐化）する可能性があります。後で修正します。
        font = ImageFont.load_default() 

        for i, img in enumerate(images):
            # 画像に書き込むための準備
            draw = ImageDraw.Draw(img)
            
            # 【仮の処理】座標 (x=100, y=100) と (x=200, y=150) に文字を書き込む
            # 実際にはここに「AIが見つけた音符のX,Y座標」が入ります
            draw.text((100, 100), "Do", font=font, fill=(255, 0, 0)) # 赤色で「Do」
            draw.text((200, 150), "Re", font=font, fill=(255, 0, 0)) # 赤色で「Re」
            
            processed_images.append(img)
            
            # 画面にもプレビュー表示
            st.image(img, caption=f"{i+1}ページ目 (処理後)", use_column_width=True)

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
