import st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 高精度・画像処理ユーティリティ関数群
# ==========================================

def deskew(pil_img):
    """画像のわずかな傾きを補正して、五線を水平にする"""
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # 二値化（背景を黒、コンテンツを白にする）
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # コンテンツの輪郭から傾き角度を計算
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # 回転を実行
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(rotated)

def detect_staff_lines_precise(pil_img):
    """五線を高精度に検出し、そのY座標を返す"""
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 横方向のオープニングで、五線だけを抽出
    # 五線の間隔を推定
    horizontal_sum = np.sum(thresh, axis=1)
    peaks = np.where(horizontal_sum > (np.max(horizontal_sum) * 0.5))[0]
    if len(peaks) < 2: return [], gray # 五線が見つからない
    staff_space = np.median(np.diff(peaks))
    
    # 五線間隔の2倍の長さの横棒フィルター
    kernel_len = int(staff_space * 2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # 抽出された五線画像から、Y座標を正確に取得
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_y_coords = []
    width = thresh.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width // 2: # 横幅が十分に長いもの
            staff_y_coords.append(int(y + h // 2)) # 中心Y座標

    staff_y_coords.sort() # 上から順に並べる
    return staff_y_coords, gray

def non_max_suppression(boxes, overlapThresh):
    """カスタムNon-Maximum Suppression実装
       近くに重なり合っている検出枠を、1つにまとめる
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def detect_note_heads_precise(gray_img, staff_space, user_threshold):
    """五線間隔から音符サイズを自動推定し、NMSで重複を除去して符頭の真の中心を検出する"""
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 五線間隔から音符（符頭）のサイズを推定
    # 符頭は通常、五線間隔とほぼ同じか少し小さい。
    note_w = int(staff_space * 1.2)
    note_h = int(staff_space * 0.9)
    
    # テンプレートを作成（黒塗りの傾いた楕円）
    template_w = note_w
    template_h = note_h
    template = np.zeros((template_h + 10, template_w + 10), dtype=np.uint8)
    center = ((template_w + 10) // 2, (template_h + 10) // 2)
    axes = (template_w // 2, template_h // 2)
    # 少し傾ける
    cv2.ellipse(template, center, axes, -20, 0, 360, 255, -1)

    # テンプレートマッチング
    result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= user_threshold)

    rectangles = []
    for pt in zip(*locations[::-1]):
        rect_data = [int(pt[0]), int(pt[1]), int(pt[0] + template_w), int(pt[1] + template_h)]
        rectangles.append(rect_data)

    if not rectangles:
        return []

    # 重複を除去する Non-Maximum Suppression (NMS)
    boxes = np.array(rectangles)
    picked_boxes = non_max_suppression(boxes, overlapThresh=0.3)

    return picked_boxes

def calculate_pitch(note_center_y, staves):
    """音符のY座標から音階を計算する"""
    # ト音記号の音階マッピング（第5線基準）
    pitch_names = {
        -2: "ラ", -1: "ソ", 0: "ファ", 1: "ミ", 2: "レ", 3: "ド",
        4: "シ", 5: "ラ", 6: "ソ", 7: "ファ", 8: "ミ", 9: "レ",
        10: "ド", 11: "シ", 12: "ラ", 13: "ソ"
    }
    
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
        return None
        
    return pitch_names.get(steps_down, "?")

# ==========================================
# 解析マスター関数 V2 (高精度版)
# ==========================================

def analyze_score_v2(pil_img, user_threshold):
    """高精度版解析関数"""
    
    # 0. 前処理：傾き補正
    deskewed_pil = deskew(pil_img)
    
    # 1. 五線検出の強化
    staff_y_coords, gray_img = detect_staff_lines_precise(deskewed_pil)
    
    if not staff_y_coords or len(staff_y_coords) < 5:
        return deskewed_pil

    # 五線間隔を計算
    staff_space = np.median(np.diff(staff_y_coords))
    
    # 2. 音符検出の強化（自動サイズ推定、NMS）
    picked_boxes = detect_note_heads_precise(gray_img, staff_space, user_threshold)

    # 3. 五線を段（5本線）にグループ化
    staves = []
    for i in range(0, len(staff_y_coords) - 4):
        # 5本の線の間隔が広すぎないかチェックして段として登録
        if staff_y_coords[i+4] - staff_y_coords[i] < staff_space * 5: 
            staves.append(staff_y_coords[i:i+5])
            
    if not staves:
        return deskewed_pil

    # 4. 音階計算と書き込み
    # 綺麗な文字を書き込むために、Pillowの画像に戻す
    result_pil = deskewed_pil.copy()
    result_pil = result_pil.convert("RGB")
    draw = ImageDraw.Draw(result_pil)
    
    # 日本語フォントの読み込み
    try:
        # 音符のサイズ（符頭の縦幅）に基づいてフォントサイズを決定
        note_h = int(staff_space * 0.9)
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(16, note_h))
    except IOError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2) in picked_boxes:
        # 音符の中心のY座標（picked_boxesはNMS済み）
        note_center_y = int((y1 + y2) / 2)
        template_h = y2 - y1
        
        # 音階計算
        doremi = calculate_pitch(note_center_y, staves)
        
        if doremi:
            # 枠を描画（確認用：緑色）
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            # ドレミを書き込む
            draw.text((x1, y1 - template_h - 10), doremi, font=font, fill=(255, 0, 0))

    return result_pil

# ==========================================
# Streamlit アプリケーション UI
# ==========================================

# ページの設定
st.set_page_config(page_title="ドレミ自動付与ツール V2", layout="centered")

st.title("🎼 楽譜ドレミ自動付与ツール V2")
st.write("PDFの楽譜をアップロードすると、自動で音階を解析します（高精度版）。")

# --- UI設定：サイドバー ---
st.sidebar.header("⚙️ 検出パラメータの調整")
st.sidebar.write("プレビューを見ながら、音符が綺麗に囲まれるように感度を調整してください。")

# 感度（閾値）のスライダーだけを残す
ui_threshold = st.sidebar.slider("検出感度 (低いほど多く検出)", 0.40, 0.95, 0.65, 0.01)
# 音符サイズの自動推定をユーザーに伝える
st.sidebar.info("音符のサイズは楽譜の五線間隔から自動で推定されます。")

# 1. ファイルアップロード機能
uploaded_file = st.file_uploader("PDF形式の楽譜を選択してください", type=["pdf"])

if uploaded_file is not None:
    st.success("ファイルを受け取りました！解析を開始します...")

    # 2. PDFを表示用に画像変換
    # ※ poppler のインストールが必要です
    images = convert_from_bytes(uploaded_file.read())
    
    # 3. 解析ロジックの呼び出し
    with st.spinner('音符を解析中...'):
        processed_images = []
        
        for i, img in enumerate(images):
            # 高精度版解析関数を呼び出す
            result_img = analyze_score_v2(img, ui_threshold)
            
            processed_images.append(result_img)
            # 画面にもプレビュー表示
            st.image(result_img, caption=f"{i+1}ページ目", use_column_width=True)

    st.success("処理が完了しました！")

    # 4. 処理結果をPDFとしてまとめる
    pdf_byte_arr = io.BytesIO()
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
