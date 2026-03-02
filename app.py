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
    """画像のわずかな傾きを補正"""
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # 適応的二値化で文字や線を浮き立たせる
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0: return pil_img
    
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
    """五線を高精度に検出し、そのY座標を返す"""
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # ガウシアンブラでノイズ低減
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 適応的二値化（ここが重要：段ごとの濃淡に対応）
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # 横方向の線だけを抽出
    width = thresh.shape[1]
    # カーネルを少し短くして、端が切れている五線も拾えるようにする
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 40, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # 行ごとの白画素の密度を計算
    line_sums = np.sum(detected_lines, axis=1)
    line_threshold = np.max(line_sums) * 0.35 # 閾値を少し下げる
    
    line_indices = np.where(line_sums > line_threshold)[0]
    
    if len(line_indices) == 0:
        return [], gray

    # 連続している行を1本の線にまとめる
    staff_y_coords = []
    if len(line_indices) > 0:
        current_group = [line_indices[0]]
        for i in range(1, len(line_indices)):
            if line_indices[i] <= line_indices[i-1] + 2: # 2ピクセル以内は同じ線
                current_group.append(line_indices[i])
            else:
                staff_y_coords.append(int(np.mean(current_group)))
                current_group = [line_indices[i]]
        staff_y_coords.append(int(np.mean(current_group)))

    return staff_y_coords, gray

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0: return []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")

    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def detect_note_heads_precise(gray_img, staff_space, user_threshold):
    # 二値化
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # 音符頭の形に近いカーネルでオープニング処理
    k_size = max(2, int(staff_space * 0.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # テンプレートの作成（少し楕円形）
    note_w = int(staff_space * 1.1)
    note_h = int(staff_space * 0.9)
    template = np.zeros((note_h, note_w), dtype=np.uint8)
    cv2.ellipse(template, (note_w//2, note_h//2), (note_w//2, note_h//2), 0, 0, 360, 255, -1)

    result = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= user_threshold)

    rectangles = []
    for pt in zip(*locations[::-1]):
        rectangles.append([pt[0], pt[1], pt[0] + note_w, pt[1] + note_h])

    if not rectangles: return []
    picked_boxes = non_max_suppression(np.array(rectangles), 0.3)
    
    final_boxes = []
    for (x1, y1, x2, y2) in picked_boxes:
        roi = thresh[y1:y2, x1:x2]
        if roi.size == 0: continue
        fill_ratio = np.count_nonzero(roi) / roi.size
        if 0.35 < fill_ratio < 0.95: # フィルタリング条件を少し緩和
            final_boxes.append([x1, y1, x2, y2])
    return final_boxes

def calculate_pitch(note_center_y, staff_lines, clef):
    """1つの段（5本線）に対する音階計算"""
    top_line_y = staff_lines[0]
    bottom_line_y = staff_lines[4]
    
    # 線と線の間隔（1ステップは間隔の半分）
    line_gap = (bottom_line_y - top_line_y) / 4.0
    step_height = line_gap / 2.0
    
    # 第5線（一番上の線）からの差分ステップ数
    steps_from_top = round((note_center_y - top_line_y) / step_height)
    
    if clef == "treble":
        # 0=ファ(第5線), 1=ミ(第4間), 2=レ(第4線)...
        pitch_names = {
            -6:"ミ", -5:"レ", -4:"ド", -3:"シ", -2:"ラ", -1:"ソ",
            0:"ファ", 1:"ミ", 2:"レ", 3:"ド", 4:"シ", 5:"ラ", 6:"ソ", 7:"ファ", 8:"ミ",
            9:"レ", 10:"ド", 11:"シ", 12:"ラ", 13:"ソ", 14:"ファ"
        }
    else:
        # 0=ラ(第5線), 1=ソ(第4間)...
        pitch_names = {
            -6:"シ", -5:"ラ", -4:"ソ", -3:"ファ", -2:"ミ", -1:"レ",
            0:"ラ", 1:"ソ", 2:"ファ", 3:"ミ", 4:"レ", 5:"ド", 6:"シ", 7:"ラ", 8:"ソ",
            9:"ファ", 10:"ミ", 11:"レ", 12:"ド", 13:"シ", 14:"ラ"
        }
    return pitch_names.get(steps_from_top, "?")

# ==========================================
# メイン解析ロジック
# ==========================================

def analyze_score_v2(pil_img, user_threshold):
    deskewed_pil = deskew(pil_img)
    staff_y_coords, gray_img = detect_staff_lines_precise(deskewed_pil)
    
    if len(staff_y_coords) < 5:
        return deskewed_pil

    # 五線のグループ化（5本ずつに分ける）
    staves = []
    i = 0
    while i <= len(staff_y_coords) - 5:
        # 5本選んで、その間隔が一定以内かチェック
        subset = staff_y_coords[i:i+5]
        gaps = np.diff(subset)
        if np.max(gaps) < np.median(gaps) * 1.8: # 異常に広い隙間がなければ採用
            staves.append(subset)
            i += 5
        else:
            i += 1 # 1本飛ばして次を探す

    if not staves: return deskewed_pil

    staff_space = np.median([np.mean(np.diff(s)) for s in staves])
    picked_boxes = detect_note_heads_precise(gray_img, staff_space, user_threshold)

    result_pil = deskewed_pil.convert("RGB")
    draw = ImageDraw.Draw(result_pil)
    
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(16, int(staff_space)))
    except:
        font = ImageFont.load_default()

    # 各段に音符を割り当て
    clefs = ["treble", "bass"] * (len(staves) // 2 + 1)
    
    for box in picked_boxes:
        nx, ny = (box[0]+box[2])//2, (box[1]+box[3])//2
        
        # 最も近い段を探す
        dist_to_staves = [abs(ny - np.mean(s)) for s in staves]
        best_staff_idx = np.argmin(dist_to_staves)
        
        # 段から遠すぎる音符は無視（ゴミ対策）
        if dist_to_staves[best_staff_idx] < staff_space * 8:
            pitch = calculate_pitch(ny, staves[best_staff_idx], clefs[best_staff_idx])
            
            # 描画
            draw.rectangle(box.tolist(), outline=(0, 200, 0), width=2)
            draw.text((box[0], box[1]-int(staff_space*1.2)), pitch, font=font, fill=(255, 0, 0))

    return result_pil

# --- Streamlit UI ---
st.set_page_config(page_title="楽譜ドレミ付与 V3", layout="wide")
st.title("🎼 楽譜ドレミ自動付与 V3 (右手検出強化版)")

ui_threshold = st.sidebar.slider("音符検出感度", 0.30, 0.90, 0.55, 0.05)
uploaded_file = st.file_uploader("楽譜PDFをアップロード", type=["pdf"])

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read())
    processed_images = []
    for i, img in enumerate(images):
        res = analyze_score_v2(img, ui_threshold)
        st.image(res, caption=f"Page {i+1}", use_container_width=True)
        processed_images.append(res)
    
    # PDFダウンロード
    pdf_buf = io.BytesIO()
    processed_images[0].save(pdf_buf, format='PDF', save_all=True, append_images=processed_images[1:])
    st.download_button("結果をダウンロード", data=pdf_buf.getvalue(), file_name="output.pdf")
