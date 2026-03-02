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
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return pil_img
        
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(rotated)

def detect_staff_lines_precise(pil_img):
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_sum = np.sum(thresh, axis=1)
    peaks = np.where(horizontal_sum > (np.max(horizontal_sum) * 0.25))[0]
    if len(peaks) < 2: return [], gray, 10
    
    diffs = np.diff(peaks)
    valid_diffs = diffs[diffs > 2]
    staff_space = np.median(valid_diffs) if len(valid_diffs) > 0 else np.median(diffs)
    
    kernel_len = max(5, int(staff_space * 1.5))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 3))
    horizontal_lines = cv2.dilate(horizontal_lines, bridge_kernel, iterations=1)

    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_y_coords = []
    width = thresh.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width // 8: 
            staff_y_coords.append(int(y + h // 2))

    staff_y_coords.sort()
    return staff_y_coords, gray, staff_space

def non_max_suppression(boxes, overlapThresh):
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

    k_size = max(2, int(staff_space * 0.4)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    note_w = int(staff_space * 1.2)
    note_h = int(staff_space * 0.9)
    
    pad = 2
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
        x1 = int(pt[0] + pad)
        y1 = int(pt[1] + pad)
        rectangles.append([x1, y1, x1 + template_w, y1 + template_h])

    if not rectangles:
        return []

    boxes = np.array(rectangles)
    picked_boxes = non_max_suppression(boxes, overlapThresh=0.4)

    final_boxes = []
    for (x1, y1, x2, y2) in picked_boxes:
        roi = thresh[y1:y2, x1:x2]
        if roi.size == 0: continue
        
        fill_ratio = np.count_nonzero(roi) / roi.size
        if 0.35 < fill_ratio < 0.95:
            final_boxes.append([x1, y1, x2, y2])

    return final_boxes

def calculate_pitch(note_center_y, staves, clefs):
    if not staves: return None
    
    distances = [abs(np.mean(staff) - note_center_y) for staff in staves]
    closest_idx = int(np.argmin(distances))
    closest_staff = staves[closest_idx]
    clef = clefs[closest_idx]
    
    top_line_y = closest_staff[0]
    bottom_line_y = closest_staff[4]
    step_height = (bottom_line_y - top_line_y) / 8.0 
    steps_down = round((note_center_y - top_line_y) / step_height)
    
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
    staff_y_coords, gray_img, staff_space = detect_staff_lines_precise(deskewed_pil)
    
    if not staff_y_coords or len(staff_y_coords) < 5:
        return deskewed_pil

    picked_boxes = detect_note_heads_precise(gray_img, staff_space, user_threshold)

    merged_y_coords = []
    for y in staff_y_coords:
        if not merged_y_coords:
            merged_y_coords.append(y)
        elif y - merged_y_coords[-1] < staff_space * 0.5:
            merged_y_coords[-1] = int((merged_y_coords[-1] + y) / 2)
        else:
            merged_y_coords.append(y)

    # 【変更】複雑なスコアリングを除外。等間隔の5本線をシンプルに抽出。
    possible_staves = []
    for i in range(len(merged_y_coords) - 4):
        staff_lines = merged_y_coords[i:i+5]
        gaps = np.diff(staff_lines)
        
        # 連続する5本の線の間隔が、すべて許容範囲内(0.7〜1.3倍)なら五線の候補とする
        if np.all((gaps > staff_space * 0.7) & (gaps < staff_space * 1.3)):
            possible_staves.append(staff_lines)
            
    staves = []
    for staff_lines in possible_staves:
        if not staves:
            staves.append(staff_lines)
        else:
            # 既に確定した段とY座標が被っていなければ追加（重複登録を防ぐだけ）
            if staff_lines[0] > staves[-1][4] + staff_space * 2:
                staves.append(staff_lines)

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

st.set_page_config(page_title="ドレミ自動付与ツール V2", layout="centered")
st.title("🎼 楽譜ドレミ自動付与ツール V2")
st.write("PDFの楽譜をアップロードすると、自動で音階を解析します（高精度版）。")

st.sidebar.header("⚙️ 検出パラメータの調整")
st.sidebar.write("プレビューを見ながら、音符が綺麗に囲まれるように感度を調整してください。")
ui_threshold = st.sidebar.slider("検出感度 (低いほど多く検出)", 0.40, 0.95, 0.65, 0.01)
st.sidebar.info("音符のサイズは楽譜の五線間隔から自動で推定されます。")

uploaded_file = st.file_uploader("PDF形式の楽譜を選択してください", type=["pdf"])

if uploaded_file is not None:
    st.success("ファイルを受け取りました！解析を開始します...")

    images = convert_from_bytes(uploaded_file.read())
    
    with st.spinner('音符を解析中...'):
        processed_images = []
        for i, img in enumerate(images):
            result_img = analyze_score_v2(img, ui_threshold)
            processed_images.append(result_img)
            st.image(result_img, caption=f"{i+1}ページ目", use_column_width=True)

    st.success("処理が完了しました！")

    pdf_byte_arr = io.BytesIO()
    processed_images[0].save(
        pdf_byte_arr, format='PDF', 
        save_all=True, append_images=processed_images[1:]
    )
    
    st.download_button(
        label="ドレミ付き楽譜をダウンロード",
        data=pdf_byte_arr.getvalue(),
        file_name="doremi_score.pdf",
        mime="application/pdf"
    )
