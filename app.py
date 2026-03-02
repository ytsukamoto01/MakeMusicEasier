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
    """画像の傾きを補正"""
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0: return pil_img
        
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
        
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(rotated)

def detect_staff_lines_precise(pil_img):
    """水平線抽出に特化した五線検出"""
    img_array = np.array(pil_img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array

    height, width = gray.shape
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 水平な長い直線だけを抽出（音符やスラーを消去）
    kernel_len = width // 20 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    clean_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # 直線の隙間を強力に補完
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 10, 1))
    clean_lines = cv2.dilate(clean_lines, bridge_kernel, iterations=1)

    contours, _ = cv2.findContours(clean_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_y_coords = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width // 5: # 画像幅の20%以上の線を採用
            staff_y_coords.append(int(y + h // 2))

    staff_y_coords.sort()

    # 近すぎる線のマージ
    merged_y = []
    for y in staff_y_coords:
        if not merged_y: merged_y.append(y)
        elif y - merged_y[-1] < 5: merged_y[-1] = int((merged_y[-1] + y) / 2)
        else: merged_y.append(y)

    # 五線間隔の推定
    staff_space = 10
    if len(merged_y) >= 2:
        diffs = np.diff(merged_y)
        valid_diffs = diffs[(diffs > 3) & (diffs < height // 20)]
        if len(valid_diffs) > 0: staff_space = np.median(valid_diffs)

    return merged_y, gray, staff_space

def non_max_suppression(boxes, overlapThresh):
    """重複した検出枠を統合"""
    if len(boxes) == 0: return []
    boxes = boxes.astype("float")
    pick, x1, y1, x2, y2 = [], boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        i = idxs[-1]
        pick.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[idxs[:-1]]), np.maximum(y1[i], y1[idxs[:-1]])
        xx2, yy2 = np.minimum(x2[i], x2[idxs[:-1]]), np.minimum(y2[i], y2[idxs[:-1]])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def detect_note_heads_precise(gray_img, staff_space, user_threshold):
    """音符の頭を検出"""
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(2, int(staff_space * 0.4)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    note_w, note_h, pad = int(staff_space * 1.2), int(staff_space * 0.9), 2
    template = np.zeros((note_h + pad*2, note_w + pad*2), dtype=np.uint8)
    cv2.ellipse(template, (note_w//2 + pad, note_h//2 + pad), (note_w//2, note_h//2), -20, 0, 360, 255, -1)

    result = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= user_threshold)

    rectangles = [[int(pt[0]+pad), int(pt[1]+pad), int(pt[0]+pad+note_w), int(pt[1]+pad+note_h)] for pt in zip(*locations[::-1])]
    if not rectangles: return []

    picked_boxes = non_max_suppression(np.array(rectangles), overlapThresh=0.4)
    final_boxes = []
    for (x1, y1, x2, y2) in picked_boxes:
        roi = thresh[y1:y2, x1:x2]
        if roi.size > 0 and 0.35 < (np.count_nonzero(roi) / roi.size) < 0.95:
            final_boxes.append([x1, y1, x2, y2])
    return final_boxes

def calculate_pitch(note_center_y, staves, clefs):
    """音階計算"""
    if not staves: return None
    closest_idx = int(np.argmin([abs(np.mean(s) - note_center_y) for s in staves]))
    staff, clef = staves[closest_idx], clefs[closest_idx]
    step_height = (staff[4] - staff[0]) / 8.0 
    steps_down = round((note_center_y - staff[0]) / step_height)
    
    # 音名テーブル
    names = {
        "treble": {-9:"ラ",-8:"ソ",-7:"ファ",-6:"ミ",-5:"レ",-4:"ド",-3:"シ",-2:"ラ",-1:"ソ",0:"ファ",1:"ミ",2:"レ",3:"ド",4:"シ",5:"ラ",6:"ソ",7:"ファ",8:"ミ",9:"レ",10:"ド",11:"シ",12:"ラ",13:"ソ",14:"ファ",15:"ミ",16:"レ"},
        "bass": {-9:"ド",-8:"シ",-7:"ラ",-6:"ソ",-5:"ファ",-4:"ミ",-3:"レ",-2:"ド",-1:"シ",0:"ラ",1:"ソ",2:"ファ",3:"ミ",4:"レ",5:"ド",6:"シ",7:"ラ",8:"ソ",9:"ファ",10:"ミ",11:"レ",12:"ド",13:"シ",14:"ラ",15:"ソ",16:"ファ"}
    }
    return names[clef].get(steps_down)

# ==========================================
# 解析マスター関数
# ==========================================

def analyze_score_v2(pil_img, user_threshold):
    deskewed_pil = deskew(pil_img)
    y_coords, gray, staff_space = detect_staff_lines_precise(deskewed_pil)
    if not y_coords or len(y_coords) < 5: return deskewed_pil

    picked_boxes = detect_note_heads_precise(gray, staff_space, user_threshold)

    # 五線のグルーピング
    merged_y = []
    for y in y_coords:
        if not merged_y or y - merged_y[-1] > staff_space * 0.5: merged_y.append(y)
        else: merged_y[-1] = int((merged_y[-1] + y) / 2)

    staves = []
    for i in range(len(merged_y) - 4):
        lines = merged_y[i:i+5]
        gaps = np.diff(lines)
        if np.all((gaps > staff_space * 0.7) & (gaps < staff_space * 1.3)):
            if not staves or lines[0] > staves[-1][4] + staff_space * 2: staves.append(lines)

    if not staves: return deskewed_pil

    result_pil = deskewed_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(result_pil)
    clefs = ["treble" if i % 2 == 0 else "bass" for i in range(len(staves))]
    
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(14, int(staff_space * 0.9)))
    except:
        font = ImageFont.load_default()

    staff_notes = {i: [] for i in range(len(staves))}
    for box in picked_boxes:
        y_c = int((box[1] + box[3]) / 2)
        idx = int(np.argmin([abs(np.mean(s) - y_c) for s in staves]))
        if (staves[idx][0] - staff_space*8) <= y_c <= (staves[idx][4] + staff_space*8):
            staff_notes[idx].append(box)

    for idx, notes in staff_notes.items():
        if not notes: continue
        notes.sort(key=lambda b: b[0])
        chords, current = [], []
        for b in notes:
            if not current or abs(b[0] - current[-1][0]) < staff_space * 0.6: current.append(b)
            else: chords.append(current); current = [b]
        if current: chords.append(current)

        for chord in chords:
            chord.sort(key=lambda b: b[1])
            pitches = [p for p in [calculate_pitch((b[1]+b[3])/2, staves, clefs) for b in chord] if p]
            if pitches:
                for b in chord: draw.rectangle(b, outline=(0, 255, 0), width=2)
                txt = "".join(sorted(set(pitches), key=pitches.index))
                tx, ty = chord[0][0], chord[0][1] - int(staff_space) - 10
                # 袋文字描画
                for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]: draw.text((tx+ox, ty+oy), txt, font=font, fill=(255,255,255))
                draw.text((tx, ty), txt, font=font, fill=(255,0,0))

    return result_pil

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(page_title="ドレミ自動付与ツール V2", layout="centered")
st.title("🎼 楽譜ドレミ自動付与ツール V2")
ui_threshold = st.sidebar.slider("検出感度", 0.40, 0.95, 0.65, 0.01)
uploaded_file = st.file_uploader("PDF形式の楽譜を選択", type=["pdf"])

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read())
    with st.spinner('解析中...'):
        processed = [analyze_score_v2(img, ui_threshold) for img in images]
        for i, img in enumerate(processed): st.image(img, caption=f"{i+1}P", use_column_width=True)
    
    pdf_buf = io.BytesIO()
    processed[0].save(pdf_buf, format='PDF', save_all=True, append_images=processed[1:])
    st.download_button(label="完成版PDFをダウンロード", data=pdf_buf.getvalue(), file_name="doremi_score.pdf", mime="application/pdf")
