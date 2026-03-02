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


# 引数に template_w, template_h, threshold を追加します
def detect_note_heads(pil_img, template_w, template_h, threshold):
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 渡されたサイズでテンプレートを作成
    template = np.zeros((template_h + 10, template_w + 10), dtype=np.uint8)
    center = ((template_w + 10) // 2, (template_h + 10) // 2)
    axes = (template_w // 2, template_h // 2)
    cv2.ellipse(template, center, axes, -20, 0, 360, 255, -1)

# -- 前半はそのまま --
    result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    # 1. 見つけた座標を [x, y, 横幅, 縦幅] のリストに変換する
    rectangles = []
    for pt in zip(*locations[::-1]):
        rect_data = [int(pt[0]), int(pt[1]), int(template_w), int(template_h)]
        rectangles.append(rect_data)
        # ※OpenCVの仕様で、単独の枠が消えないように同じものを2回入れるテクニックです
        rectangles.append(rect_data) 

    # 2. 重なった四角形を1つにまとめる魔法の関数！
    # groupThreshold=1 (1つ以上重なっているものを統合), eps=0.5 (まとめる距離の寛容さ)
    grouped_rects, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

    # 3. 画面確認用（元の画像に緑色の枠を描く）
    result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # まとまった後のスッキリした四角形だけを描画
    for (x, y, w, h) in grouped_rects:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return Image.fromarray(result_img)

def analyze_score(pil_img, template_w, template_h, threshold):
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # ==========================================
    # 1. 五線の検出とグループ化（クラスタリング）
    # ==========================================
    horizontal_sum = np.sum(thresh, axis=1)
    width = thresh.shape[1]
    line_threshold = width * 255 * 0.5
    raw_y_coords = np.where(horizontal_sum > line_threshold)[0]

    # 太い線を「1本のY座標」にまとめる
    lines = []
    current_cluster = []
    for y in raw_y_coords:
        if not current_cluster:
            current_cluster.append(y)
        elif y - current_cluster[-1] <= 5: # 5px以内のズレは同じ線とみなす
            current_cluster.append(y)
        else:
            lines.append(int(np.mean(current_cluster)))
            current_cluster = [y]
    if current_cluster:
        lines.append(int(np.mean(current_cluster)))

    # 5本ずつグループ化して「1つの段」にする
    staves = []
    for i in range(0, len(lines) - 4):
        # 5本の線の間隔が広すぎないかチェックして段として登録
        if lines[i+4] - lines[i] < template_h * 5: 
            staves.append(lines[i:i+5])

    # ==========================================
    # 2. 音符の検出とグループ化
    # ==========================================
    template = np.zeros((template_h + 10, template_w + 10), dtype=np.uint8)
    center = ((template_w + 10) // 2, (template_h + 10) // 2)
    axes = (template_w // 2, template_h // 2)
    cv2.ellipse(template, center, axes, -20, 0, 360, 255, -1)

    result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    rectangles = []
    for pt in zip(*locations[::-1]):
        rect_data = [int(pt[0]), int(pt[1]), int(template_w), int(template_h)]
        rectangles.append(rect_data)
        rectangles.append(rect_data)

    grouped_rects, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

    # ==========================================
    # 3. Y座標から「ドレミ」を計算して書き込む！
    # ==========================================
    result_pil = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(result_pil)
    try:
        font = ImageFont.truetype("font.ttf", max(16, template_h))
    except IOError:
        font = ImageFont.load_default()

    if not staves:
        return result_pil

    # ト音記号のドレミ配列（基準の第5線=0 から、下に何ステップ移動したか）
    pitch_names = {
        -2: "ラ", -1: "ソ", 0: "ファ", 1: "ミ", 2: "レ", 3: "ド",
        4: "シ", 5: "ラ", 6: "ソ", 7: "ファ", 8: "ミ", 9: "レ",
        10: "ド", 11: "シ", 12: "ラ", 13: "ソ"
    }

    for (x, y, w, h) in grouped_rects:
        note_center_y = y + h // 2
        
        # 一番近い「段（5本線のグループ）」を探す
        closest_staff = min(staves, key=lambda staff: abs(np.mean(staff) - note_center_y))
        
        top_line_y = closest_staff[0]
        bottom_line_y = closest_staff[4]
        
        # 五線の間隔から「1ステップ（音符1つ分の縦移動）」の高さを計算
        # 5本の線の間には4つの空間があるため、線〜線までの距離を8分割すると1ステップになります
        step_height = (bottom_line_y - top_line_y) / 8.0 
        
        # 一番上の線からの距離をステップ数に変換（四捨五入で最も近い位置に合わせる）
        steps_down = round((note_center_y - top_line_y) / step_height)
        
        # 加線が多すぎる（範囲外）ものは文字やゴミとして除外
        if steps_down < -3 or steps_down > 14:
            continue
            
        # ドレミを取得！
        doremi = pitch_names.get(steps_down, "?")
        
        # 枠と文字を描画
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
        draw.text((x, y - template_h - 10), doremi, font=font, fill=(255, 0, 0))

    return result_pil

# ページの設定
st.set_page_config(page_title="ドレミ自動付与ツール", layout="centered")

st.title("🎼 楽譜ドレミ自動付与ツール")
st.write("PDFの楽譜をアップロードすると、自動で音階を解析します。")

# --- UI設定：サイドバーにスライダーを配置 ---
st.sidebar.header("⚙️ 検出パラメータの調整")
st.sidebar.write("プレビューを見ながら、音符が綺麗に囲まれるように調整してください。")

# スライダーの作成 (ラベル, 最小値, 最大値, デフォルト値, 刻み幅)
ui_w = st.sidebar.slider("音符の横幅 (ピクセル)", 5, 50, 24, 1)
ui_h = st.sidebar.slider("音符の縦幅 (ピクセル)", 5, 50, 16, 1)
ui_threshold = st.sidebar.slider("検出感度 (低いほど多く検出)", 0.40, 0.95, 0.65, 0.01)

# 1. ファイルアップロード機能
uploaded_file = st.file_uploader("PDF形式の楽譜を選択してください", type=["pdf"])

if uploaded_file is not None:
    st.success("ファイルを受け取りました！解析を開始します...")

    # 2. PDFを表示用に画像変換 (プレビュー)
    # ※実際の運用には poppler のインストールが必要です
    images = convert_from_bytes(uploaded_file.read())
    
    for i, image in enumerate(images):
        st.image(image, caption=f"{i+1}ページ目", use_column_width=True)

    # 3. 画像への書き込み処理 のループ内を以下のように変更
    with st.spinner('音符を解析中...'):
        processed_images = []
        
        for i, img in enumerate(images):
            # 新しい合体関数を呼び出す！
            result_img = analyze_score(img, ui_w, ui_h, ui_threshold)
            
            processed_images.append(result_img)
            st.image(result_img, caption=f"{i+1}ページ目", use_column_width=True)

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
