import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 1. 画像処理ユーティリティ
# ==========================================

def get_staff_lines_by_projection(pil_img):
    """
    水平投影を使用して五線を検出する。
    """
    img_array = np.array(pil_img.convert('L'))
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    projection = np.sum(thresh, axis=1)
    
    line_candidates = []
    thresh_val = np.max(projection) * 0.5
    for i in range(1, len(projection) - 1):
        if projection[i] > thresh_val and projection[i] >= projection[i-1] and projection[i] >= projection[i+1]:
            line_candidates.append(i)

    merged_lines = []
    if line_candidates:
        curr = line_candidates[0]
        for i in range(1, len(line_candidates)):
            if line_candidates[i] - curr < 3:
                curr = (curr + line_candidates[i]) // 2
            else:
                merged_lines.append(curr)
                curr = line_candidates[i]
        merged_lines.append(curr)

    staves = []
    i = 0
    while i <= len(merged_lines) - 5:
        diffs = np.diff(merged_lines[i:i+5])
        avg_spacing = np.mean(diffs)
        if np.all(np.abs(diffs - avg_spacing) < avg_spacing * 0.4):
            # [第1線, 第2線, 第3線, 第4線, 第5線] (下から上)
            staves.append(sorted(merged_lines[i:i+5], reverse=True))
            i += 5
        else:
            i += 1
            
    return staves, avg_spacing if staves else 10

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0: return []
    boxes = boxes.astype("float")
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
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(2, int(staff_space * 0.5)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    nw, nh = int(staff_space * 1.2), int(staff_space * 0.9)
    template = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    cv2.ellipse(template, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, -1)
    
    res = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= user_threshold)
    rects = [[int(pt[0]), int(pt[1]), int(pt[0]+nw), int(pt[1]+nh)] for pt in zip(*loc[::-1])]
    
    if not rects: return []
    return non_max_suppression_fast(np.array(rects), overlapThresh=0.3)

# ==========================================
# 2. 音階判定ロジック
# ==========================================

def get_pitch_info_v3(note_y, staff_lines, clef, flats_count):
    bottom_line = staff_lines[0]
    top_line = staff_lines[4]
    step_size = abs(top_line - bottom_line) / 8.0
    steps_from_bottom = round((bottom_line - note_y) / step_size)

    if clef == "treble":
        mapping = {
            -4:"ラ", -3:"シ", -2:"ド", -1:"レ", 
            0:"ミ", 1:"ファ", 2:"ソ", 3:"ラ", 4:"シ", 5:"ド", 6:"レ", 7:"ミ", 8:"ファ",
            9:"ソ", 10:"ラ", 11:"シ", 12:"ド"
        }
    else: # bass
        mapping = {
            -2:"ミ", -1:"ファ",
            0:"ソ", 1:"ラ", 2:"シ", 3:"ド", 4:"レ", 5:"ミ", 6:"ファ", 7:"ソ", 8:"ラ",
            9:"シ", 10:"ド", 11:"レ", 12:"ミ"
        }

    base_name = mapping.get(steps_from_bottom, "")
    if not base_name: return "", False

    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    is_flat = base_name in flat_order[:flats_count]
    
    return base_name, is_flat

# ==========================================
# 3. メイン解析関数
# ==========================================

def analyze_score_final_v3(pil_img, threshold, flats_count):
    staves, space = get_staff_lines_by_projection(pil_img)
    if not staves: return pil_img

    img_gray = np.array(pil_img.convert('L'))
    notes = detect_note_heads_precise(img_gray, space, threshold)
    
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(14, int(space)))
    except: font = ImageFont.load_default()

    for box in notes:
        # 【重要】box (numpy配列) をリスト形式に変換してエラーを回避
        box_list = box.tolist()
        
        yc = (box_list[1] + box_list[3]) // 2
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        
        p_name, is_flat = get_pitch_info_v3(yc, staves[s_idx], clef, flats_count)
        
        if p_name:
            # box_list を使用して描画
            draw.rectangle(box_list, outline=(0, 255, 0), width=2)
            text_color = (0, 0, 255) if is_flat else (255, 0, 0)
            draw.text((box_list[0], box_list[1] - int(space * 1.5)), p_name, font=font, fill=text_color)

    return result

# ==========================================
# 4. Streamlit UI
# ==========================================

st.set_page_config(page_title="ドレミ付与ツール V3.1", layout="centered")
st.title("🎼 楽譜ドレミ付与ツール V3.1")
st.write("エラーを修正しました。音階を判定して表示します。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4) 
sens = st.sidebar.slider("検出感度", 0.4, 0.95, 0.65, 0.01)

up = st.file_uploader("PDF形式の楽譜をアップロードしてください", type="pdf")

if up:
    # PDFを画像に変換
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        processed_imgs = []
        for im in imgs:
            res_im = analyze_score_final_v3(im, sens, flats)
            processed_imgs.append(res_im)
            st.image(res_im, use_column_width=True)

    if processed_imgs:
        pdf_buf = io.BytesIO()
        processed_imgs[0].save(pdf_buf, format='PDF', save_all=True, append_images=processed_imgs[1:])
        st.download_button(label="完成版PDFをダウンロード", data=pdf_buf.getvalue(), file_name="doremi_score_v3.pdf", mime="application/pdf")
