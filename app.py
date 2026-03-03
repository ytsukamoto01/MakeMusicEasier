import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 楽譜解析エンジン（既存の関数をここに配置）
# ==========================================
# def detect_staff_groups_v8(pil_img): ...
# def nms_v8_strict(boxes, scores, staff_space): ...
# def detect_note_heads_v8(gray_img, staff_space, threshold_val, staves): ...
# def get_pitch_name(note_y, staff, clef, flats_count): ...

# ==========================================
# 新規: 描画専用の関数（自動検出分 ＋ ユーザークリック分）
# ==========================================
def draw_all_notes(pil_img, auto_notes, custom_clicks, staves, space, flats_count):
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(15, int(space)))
    except:
        font = ImageFont.load_default()

    # 共通の描画処理
    def draw_note_label(x, yc, is_custom=False):
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        p_name, is_flat = get_pitch_name(yc, staves[s_idx], clef, flats_count)
        
        if p_name:
            # ユーザークリックの場合は枠線の色を変える（例：オレンジ）
            box_color = (255, 165, 0) if is_custom else (0, 255, 0)
            
            # 擬似的なバウンディングボックスを作成
            half_w, half_h = int(space * 0.6), int(space * 0.5)
            b = [x - half_w, yc - half_h, x + half_w, yc + half_h]
            
            draw.rectangle(b, outline=box_color, width=2)
            text_color = (0, 0, 255) if is_flat else (255, 0, 0)
            draw.text((b[0], b[1] - int(space * 1.6)), p_name, font=font, fill=text_color)

    # 1. 自動検出された音符を描画
    for box in auto_notes:
        b = box.tolist()
        yc = (b[1] + b[3]) // 2
        cx = (b[0] + b[2]) // 2
        draw_note_label(cx, yc, is_custom=False)

    # 2. ユーザーがクリックした音符を描画
    for (cx, yc) in custom_clicks:
        draw_note_label(cx, yc, is_custom=True)

    return result

# ==========================================
# 新規: キャッシュを使った重い処理の分離
# ==========================================
@st.cache_data(show_spinner=False)
def process_pdf_and_detect(pdf_bytes, internal_threshold):
    """PDF変換と自動検出をキャッシュし、クリックのたびに再実行されるのを防ぐ"""
    imgs = convert_from_bytes(pdf_bytes)
    page_data = []
    
    for img in imgs:
        staves, space = detect_staff_groups_v8(img)
        img_gray = np.array(img.convert('L'))
        
        if staves:
            notes = detect_note_heads_v8(img_gray, space, internal_threshold, staves)
        else:
            notes = []
            
        page_data.append({
            "image": img,
            "staves": staves,
            "space": space,
            "notes": notes
        })
    return page_data

# ==========================================
# 2. Streamlit UI
# ==========================================
st.set_page_config(page_title="ドレミ付与 V8", layout="centered")
st.title("🎼 ドレミ付与ツール V8 (スマートノイズ除去モード)")
st.write("「細い線」だけを溶かす処理により、音符の形を崩さずに記号や五線を綺麗に除外します。")
st.info("💡 **Tips:** 検出漏れがあった場合、画像上の音符（おたまじゃくし）を直接クリックすると、その位置の音階を自動認識して追加します！")

# セッションステートの初期化（ユーザークリック座標の保存用）
if "custom_clicks" not in st.session_state:
    st.session_state.custom_clicks = {} # {page_index: [(x, y), ...]}

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4)
ui_sens = st.sidebar.slider("検出感度（1〜100：大きいほどたくさん検出）", min_value=1, max_value=100, value=50, step=1)
internal_threshold = 0.85 - (ui_sens / 100.0) * 0.40

up = st.file_uploader("PDFをアップロード", type="pdf")

if up:
    pdf_bytes = up.read()
    
    with st.spinner('解析中...'):
        # キャッシュされた解析処理を実行
        pages = process_pdf_and_detect(pdf_bytes, internal_threshold)
    
    # 各ページを描画
    for i, page in enumerate(pages):
        st.write(f"### ページ {i + 1}")
        
        if page["staves"]:
            # このページのカスタムクリック履歴を取得
            clicks = st.session_state.custom_clicks.get(i, [])
            
            # 描画処理（自動検出 ＋ カスタムクリック）
            result_img = draw_all_notes(
                page["image"], 
                page["notes"], 
                clicks, 
                page["staves"], 
                page["space"], 
                flats
            )
            
            # 画像を表示し、クリック座標を取得
            value = streamlit_image_coordinates(result_img, key=f"img_coords_{i}")
            
            # 画像がクリックされた場合の処理
            if value is not None:
                clicked_x, clicked_y = value["x"], value["y"]
                
                # 同じ座標が既に登録されていないかチェック（連続クリック防止）
                if (clicked_x, clicked_y) not in clicks:
                    if i not in st.session_state.custom_clicks:
                        st.session_state.custom_clicks[i] = []
                    
                    # クリック座標を保存し、アプリを再実行して画面を更新
                    st.session_state.custom_clicks[i].append((clicked_x, clicked_y))
                    st.rerun()
                    
        else:
            # 五線が検出されなかった場合はそのまま表示
            st.image(page["image"], use_column_width=True)

    # やり直し機能
    if st.button("手動で追加した音符をリセット"):
        st.session_state.custom_clicks = {}
        st.rerun()
