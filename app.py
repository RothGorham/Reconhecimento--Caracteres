import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import io
import base64

# Arquitetura da rede neural convolucional para classificar digitos
class RedeSimples(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extrator = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classificador(self.extrator(x))


# Carrega o modelo treinado a partir do arquivo .pth
@st.cache_resource
def carregar_modelo(caminho):
    modelo = RedeSimples()
    estado = torch.load(caminho, map_location="cpu")
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo


# Preprocessa imagem enviada por upload (detecta fundo, binariza, centraliza)
def preprocessar(img: Image.Image):
    from scipy import ndimage

    img = img.convert("L")
    arr = np.array(img, dtype=np.float32)

    altura, largura = arr.shape
    canto_size = max(3, min(10, altura // 4, largura // 4))

    cantos = np.concatenate([
        arr[:canto_size, :canto_size].flatten(),
        arr[:canto_size, -canto_size:].flatten(),
        arr[-canto_size:, :canto_size].flatten(),
        arr[-canto_size:, -canto_size:].flatten()
    ])

    valor_fundo = np.median(cantos)
    if valor_fundo > 127:
        arr = 255.0 - arr

    sigma = max(1.0, min(altura, largura) / 60.0)
    arr = ndimage.gaussian_filter(arr, sigma=sigma)

    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(arr_uint8, bins=256, range=(0, 256))
    total = arr_uint8.size
    soma_total = np.sum(np.arange(256) * hist)
    soma_bg = 0.0
    peso_bg = 0
    max_variancia = 0.0
    melhor_threshold = 0

    for t in range(256):
        peso_bg += hist[t]
        if peso_bg == 0:
            continue
        peso_fg = total - peso_bg
        if peso_fg == 0:
            break
        soma_bg += t * hist[t]
        media_bg = soma_bg / peso_bg
        media_fg = (soma_total - soma_bg) / peso_fg
        variancia = peso_bg * peso_fg * (media_bg - media_fg) ** 2
        if variancia > max_variancia:
            max_variancia = variancia
            melhor_threshold = t

    arr_bin = np.where(arr >= melhor_threshold, 255.0, 0.0)

    kernel_size = max(2, min(altura, largura) // 50)
    struct = np.ones((kernel_size, kernel_size))
    arr_bin = ndimage.binary_closing(arr_bin > 127, structure=struct).astype(np.float32) * 255.0

    img_proc = Image.fromarray(arr_bin.astype(np.uint8))

    bbox = img_proc.getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        margem = max(2, int(0.05 * max(x1 - x0, y1 - y0)))
        x0 = max(0, x0 - margem)
        y0 = max(0, y0 - margem)
        x1 = min(largura, x1 + margem)
        y1 = min(altura, y1 + margem)
        img_proc = img_proc.crop((x0, y0, x1, y1))

    w, h = img_proc.size
    if w == 0 or h == 0:
        w, h = 1, 1
    escala = 20.0 / max(w, h)
    new_w = max(1, int(w * escala))
    new_h = max(1, int(h * escala))
    img_proc = img_proc.resize((new_w, new_h), Image.LANCZOS)

    arr_proc = np.array(img_proc, dtype=np.float32)
    if arr_proc.sum() > 0:
        cy, cx = ndimage.center_of_mass(arr_proc)
    else:
        cy, cx = new_h / 2.0, new_w / 2.0

    offset_x = int(round(14.0 - cx))
    offset_y = int(round(14.0 - cy))

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img_proc, (offset_x, offset_y))

    arr_final = np.array(canvas, dtype=np.float32) / 255.0
    arr_final = (arr_final - 0.1307) / 0.3081
    tensor = torch.tensor(arr_final).unsqueeze(0).unsqueeze(0)
    return tensor, canvas


# Preprocessa imagem desenhada no canvas (ja e branco no preto)
def preprocessar_canvas(img: Image.Image):
    from scipy import ndimage

    img = img.convert("L")

    bbox = img.getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        margem = max(2, int(0.05 * max(x1 - x0, y1 - y0)))
        w_img, h_img = img.size
        x0 = max(0, x0 - margem)
        y0 = max(0, y0 - margem)
        x1 = min(w_img, x1 + margem)
        y1 = min(h_img, y1 + margem)
        img = img.crop((x0, y0, x1, y1))

    w, h = img.size
    if w == 0 or h == 0:
        w, h = 1, 1
    escala = 20.0 / max(w, h)
    new_w = max(1, int(w * escala))
    new_h = max(1, int(h * escala))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    arr_proc = np.array(img, dtype=np.float32)

    canvas = Image.new("L", (28, 28), 0)
    if arr_proc.sum() > 0:
        cy, cx = ndimage.center_of_mass(arr_proc)
        offset_x = int(round(14.0 - cx))
        offset_y = int(round(14.0 - cy))
    else:
        offset_x, offset_y = 4, 4

    canvas.paste(img, (offset_x, offset_y))

    arr_final = np.array(canvas, dtype=np.float32) / 255.0
    arr_final = (arr_final - 0.1307) / 0.3081
    tensor = torch.tensor(arr_final).unsqueeze(0).unsqueeze(0)
    return tensor, canvas


# Faz a predicao do digito e retorna as probabilidades de cada classe
def prever(modelo, tensor):
    with torch.no_grad():
        saida = modelo(tensor)
        probs = torch.softmax(saida, dim=1)[0]
        pred = probs.argmax().item()
    return pred, probs.numpy()


st.set_page_config(page_title="Reconhecimento de Digitos", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'Space Mono', monospace; }
h1 { font-family: 'Syne', sans-serif; font-size: 2.4rem !important; letter-spacing: -1px; color: #f0f0f0; }
.digit-box {
    background: #111; border: 2px solid #333; border-radius: 16px;
    padding: 2rem; text-align: center; margin: 1rem 0;
}
.big-pred {
    font-family: 'Syne', sans-serif; font-size: 7rem;
    font-weight: 800; color: #00e5ff; line-height: 1;
}
.bar-wrap { display: flex; align-items: center; gap: 0.6rem; margin: 3px 0; }
.bar-digit { width: 16px; color: #ccc; text-align: right; font-size: 0.85rem; }
.bar-bg    { flex: 1; background: #222; border-radius: 4px; height: 14px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 4px; background: #00e5ff; transition: width .4s; }
.bar-pct   { width: 44px; color: #aaa; font-size: 0.78rem; text-align: right; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("Reconhecimento de Digitos")

with st.sidebar:
    st.header("Configuracao")
    arquivo_modelo = st.file_uploader("melhor_modelo.pth", type=["pth"])

    modelo = None
    if arquivo_modelo:
        with open("modelo_temp.pth", "wb") as f:
            f.write(arquivo_modelo.read())
        try:
            modelo = carregar_modelo("modelo_temp.pth")
            st.success("Modelo carregado")
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
    else:
        st.info("Faca upload do arquivo .pth para comecar.")

    st.markdown("**Modelo:** RedeSimples  \n**Dataset:** MNIST  \n**Acuracia:** 99.21%")
    st.markdown("**Dicas:**\n- Digito centralizado\n- Fundo uniforme\n- Traco grosso\n- Sem outros elementos")

tab_upload, tab_desenho = st.tabs(["Upload de imagem", "Desenhar"])
imagem_entrada = None
fonte_canvas = False

with tab_upload:
    arq = st.file_uploader("Escolha uma imagem com um digito", type=["png","jpg","jpeg","bmp","webp"])
    if arq:
        imagem_entrada = Image.open(arq)

with tab_desenho:
    st.markdown("Desenhe um digito no canvas abaixo e clique **Reconhecer**:")

    canvas_html = """
    <style>
        #canvas-container {
            display: flex; flex-direction: column; align-items: center; gap: 12px;
        }
        #draw-canvas {
            border: 2px solid #555; border-radius: 8px; cursor: crosshair;
            background: #000; touch-action: none;
        }
        .canvas-btn {
            padding: 8px 24px; border: 1px solid #555; border-radius: 6px;
            background: #222; color: #ccc; cursor: pointer; font-family: 'Space Mono', monospace;
            font-size: 0.85rem; transition: background 0.2s;
        }
        .canvas-btn:hover { background: #333; }
        .canvas-btn.primary { background: #00e5ff; color: #111; border-color: #00e5ff; font-weight: 700; }
        .canvas-btn.primary:hover { background: #00b8d4; }
        .btn-row { display: flex; gap: 10px; }
    </style>
    <div id="canvas-container">
        <canvas id="draw-canvas" width="280" height="280"></canvas>
        <div class="btn-row">
            <button class="canvas-btn" onclick="clearCanvas()">Limpar</button>
            <button class="canvas-btn primary" onclick="sendCanvas()">Reconhecer</button>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('draw-canvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        let lastX = 0, lastY = 0;

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, 280, 280);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return [clientX - rect.left, clientY - rect.top];
        }

        canvas.addEventListener('mousedown', (e) => { drawing = true; [lastX, lastY] = getPos(e); });
        canvas.addEventListener('mousemove', (e) => {
            if (!drawing) return;
            const [x, y] = getPos(e);
            ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke();
            [lastX, lastY] = [x, y];
        });
        canvas.addEventListener('mouseup', () => { drawing = false; });
        canvas.addEventListener('mouseleave', () => { drawing = false; });

        canvas.addEventListener('touchstart', (e) => { e.preventDefault(); drawing = true; [lastX, lastY] = getPos(e); });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (!drawing) return;
            const [x, y] = getPos(e);
            ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(x, y); ctx.stroke();
            [lastX, lastY] = [x, y];
        });
        canvas.addEventListener('touchend', () => { drawing = false; });

        function clearCanvas() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, 280, 280);
        }

        function sendCanvas() {
            const dataUrl = canvas.toDataURL('image/png');
            const base64 = dataUrl.split(',')[1];
            const url = new URL(window.parent.location);
            url.searchParams.set('canvas_data', base64);
            window.parent.history.replaceState({}, '', url);
            window.parent.postMessage({stCommands: [{type: "rerun"}]}, '*');
            setTimeout(() => {
                const btns = window.parent.document.querySelectorAll('button');
                btns.forEach(b => {
                    if (b.textContent.trim() === 'Processar desenho') b.click();
                });
            }, 200);
        }
    </script>
    """
    components.html(canvas_html, height=350)

    if st.button("Processar desenho", key="btn_processar"):
        pass

    params = st.query_params
    canvas_data = params.get("canvas_data", None)
    if canvas_data:
        try:
            img_bytes = base64.b64decode(canvas_data)
            imagem_entrada = Image.open(io.BytesIO(img_bytes))
            fonte_canvas = True
        except Exception:
            pass

if imagem_entrada and modelo:
    col1, col2 = st.columns([1, 1])

    with col1:
        if not fonte_canvas:
            st.image(imagem_entrada, caption="Imagem original", use_container_width=True)

    if fonte_canvas:
        tensor, preview = preprocessar_canvas(imagem_entrada)
    else:
        tensor, preview = preprocessar(imagem_entrada)

    with col2:
        st.image(
            preview.resize((140, 140), Image.NEAREST),
            caption="Preview 28x28 (entrada da rede)",
            use_container_width=False,
        )

    pred, probs = prever(modelo, tensor)
    confianca = probs[pred] * 100

    st.markdown(f"""
    <div class="digit-box">
        <div style="color:#aaa;font-size:.9rem;margin-bottom:.4rem">DIGITO RECONHECIDO</div>
        <div class="big-pred">{pred}</div>
        <div style="color:#aaa;margin-top:.5rem">confianca: <b style="color:#00e5ff">{confianca:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Probabilidades por classe:**")
    bars_html = ""
    for i, p in enumerate(probs):
        pct = p * 100
        cor = "#00e5ff" if i == pred else "#444"
        bars_html += f"""
        <div class="bar-wrap">
          <div class="bar-digit">{i}</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{cor}"></div></div>
          <div class="bar-pct">{pct:.1f}%</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

elif imagem_entrada and not modelo:
    st.warning("Carregue o melhor_modelo.pth na barra lateral primeiro.")
