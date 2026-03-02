import streamlit as st
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
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 削りすぎないよう少し優しめに設定
    k_size = max(2, int(staff_space * 0.45)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    note_w = int(staff_space * 1.2)
    note_h = int(staff_space * 0.9)
    
    # 【大バグ修正！】パッド（余白）のズレを修正
    pad = 5
    template_w = note_w
    template_h = note_h
    template = np.zeros((template_h + pad*2, template_w + pad*2), dtype=np.uint8)
    center = (template_w // 2 + pad, template_h // 2 + pad)
    axes = (template_w // 2, template_h // 2)
    cv2.ellipse(template, center, axes, -20, 0, 360, 255, -1)

    result = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= user_threshold)

    rectangles = []
    for pt in zip(*locations[::-1]):
        # 余白(pad)の分だけ座標を補正し、純粋な音符だけを囲む！
        x1 = int(pt[0] + pad)
        y1 = int(pt[1] + pad)
        rect_data = [x1, y1, x1 + template_w, y1 + template_h]
        rectangles.append(rect_data)

    if not rectangles:
        return []

    boxes = np.array(rectangles)
    picked_boxes = non_max_suppression(boxes, overlapThresh=0.3)

    final_boxes = []
    for (x1, y1, x2, y2) in picked_boxes:
        roi = thresh[y1:y2, x1:x2]
        if roi.size == 0: continue
        
        fill_ratio = np.count_nonzero(roi) / roi.size
        
        # 【新フィルター】スカスカなノイズ(40%未満)と、真っ黒な連桁(90%以上)の両方をカット！
        if 0.40 < fill_ratio < 0.90:
            final_boxes.append([x1, y1, x2, y2])

    return final_boxes

def calculate_pitch(note_center_y, staves, clefs):
    """音符のY座標と音部記号（ト音/ヘ音）から音階を正しく計算する"""
    if not staves: return None
    
    distances = [abs(np.mean(staff) - note_center_y) for staff in staves]
    closest_idx = int(np.argmin(distances))
    closest_staff = staves[closest_idx]
    clef = clefs[closest_idx]
    
    top_line_y = closest_staff[0]
    bottom_line_y = closest_staff[4]
    step_height = (bottom_line_y - top_line_y) / 8.0 
    steps_down = round((note_center_y - top_line_y) / step_height)
    
    # 【修正2】ショパンの超高音・超低音（加線）に対応できるよう辞書を大幅拡大！
    if clef == "treble":
        pitch_names = {
            -9: "ラ", -8: "ソ", -7: "ファ", -6: "ミ", -5: "レ", -4: "ド", -3: "シ", -2: "ラ", -1: "ソ", 
            0: "ファ", 1: "ミ", 2: "レ", 3: "ド", 4: "シ", 5: "ラ", 6: "ソ", 7: "ファ", 
            8: "ミ", 9: "レ", 10: "ド", 11: "シ", 12: "ラ", 13: "ソ", 14: "ファ", 15: "ミ", 16: "レ"
        }
    else:
        pitch_names = {
            -9: "ド", -8: "シ", -7: "ラ", -6: "ソ", -5: "ファ", -4: "ミ", -3: "レ", -2: "ド", -1: "シ", 
            0: "ラ", 1: "ソ", 2: "ファ", 3: "ミ", 4: "レ", 5: "ド", 6: "シ", 7: "ラ", 
            8: "ソ", 9: "ファ", 10: "ミ", 11: "レ", 12: "ド", 13: "シ", 14: "ラ", 15: "ソ", 16: "ファ"
        }
        
    return pitch_names.get(steps_down, None)

# ==========================================
# 解析マスター関数 V2 (高精度版)
# ==========================================

def analyze_score_v2(pil_img, user_threshold):
    deskewed_pil = deskew(pil_img)
    staff_y_coords, gray_img = detect_staff_lines_precise(deskewed_pil)
    
    if not staff_y_coords or len(staff_y_coords) < 5:
        return deskewed_pil

    staff_space = np.median(np.diff(staff_y_coords))
    picked_boxes = detect_note_heads_precise(gray_img, staff_space, user_threshold)

    # 【大革命】どんなにノイズ線が混じっていても、完璧な5本線の組み合わせだけを抽出し続ける探索ロジック
    staves = []
    for i in range(len(staff_y_coords)):
        y1 = staff_y_coords[i]
        # すでに登録済みの段と被る場合はスキップ
        if staves and y1 < staves[-1][4] + staff_space * 2:
            continue
        
        current_staff = [y1]
        last_y = y1
        for j in range(i+1, len(staff_y_coords)):
            yj = staff_y_coords[j]
            # 次の線が、五線間隔の許容範囲内なら追加（ノイズ線は無視される）
            if staff_space * 0.6 < (yj - last_y) < staff_space * 1.4:
                current_staff.append(yj)
                last_y = yj
            if len(current_staff) == 5:
                break
        
        if len(current_staff) == 5:
            staves.append(current_staff)
            
    if not staves:
        return deskewed_pil

    clefs = ["treble" if i % 2 == 0 else "bass" for i in range(len(staves))]

    result_pil = deskewed_pil.copy()
    result_pil = result_pil.convert("RGB")
    draw = ImageDraw.Draw(result_pil)
    
    try:
        note_h = int(staff_space * 0.9)
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(14, note_h))
    except IOError:
        font = ImageFont.load_default()

    staff_notes = {i: [] for i in range(len(staves))}
    
    # 【修正】ショパンの超高音に対応するため、マージンをさらに拡大（8加線分まで許可）
    margin = staff_space * 8 
    
    img_width = deskewed_pil.size[0]
    ignore_x_zone = int(img_width * 0.08)
    
    for box in picked_boxes:
        if box[0] < ignore_x_zone:
            continue
            
        note_center_y = int((box[1] + box[3]) / 2)
        distances = [abs(np.mean(staff) - note_center_y) for staff in staves]
        closest_idx = int(np.argmin(distances))
        
        closest_staff = staves[closest_idx]
        if (closest_staff[0] - margin) <= note_center_y <= (closest_staff[4] + margin):
            staff_notes[closest_idx].append(box)

    for staff_idx, notes in staff_notes.items():
        if not notes: continue
            
        notes = sorted(notes, key=lambda b: b[0])
        chords = []
        current_chord = []
        
        for box in notes:
            if not current_chord:
                current_chord.append(box)
            else:
                if abs(box[0] - current_chord[-1][0]) < (staff_space * 0.6):
                    current_chord.append(box)
                else:
                    chords.append(current_chord)
                    current_chord = [box]
        if current_chord:
            chords.append(current_chord)

        for chord in chords:
            chord.sort(key=lambda b: b[1])
            pitches = []
            for (x1, y1, x2, y2) in chord:
                note_center_y = int((y1 + y2) / 2)
                doremi = calculate_pitch(note_center_y, staves, clefs)
                if doremi:
                    pitches.append(doremi)
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            
            if pitches:
                unique_pitches = []
                for p in pitches:
                    if not unique_pitches or unique_pitches[-1] != p:
                        unique_pitches.append(p)
                
                chord_text = "".join(unique_pitches)
                
                top_box = chord[0]
                tx = top_box[0] - (len(chord_text) * 2) 
                ty = top_box[1] - note_h - 15
                
                draw.text((tx-1, ty), chord_text, font=font, fill=(255, 255, 255))
                draw.text((tx+1, ty), chord_text, font=font, fill=(255, 255, 255))
                draw.text((tx, ty-1), chord_text, font=font, fill=(255, 255, 255))
                draw.text((tx, ty+1), chord_text, font=font, fill=(255, 255, 255))
                draw.text((tx, ty), chord_text, font=font, fill=(255, 0, 0))

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
