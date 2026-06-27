"""Kitchen-themed Gradio chat interface with token streaming."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import gradio as gr

# Allow running as `python frontend/app.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import stream_query  # noqa: E402


def _warmup_backend() -> None:
    """Load SQLite + Chroma once at startup."""
    try:
        from backend.config import CHROMA_DIR, SQLITE_PATH
        from backend.database import get_vectorstore, reset_vectorstore
        from backend.recipe_db import get_connection

        reset_vectorstore()
        get_connection()
        get_vectorstore()
        print(f"✓ SQLite lista ({SQLITE_PATH})")
        print(f"✓ Chroma lista ({CHROMA_DIR})")
    except Exception as exc:
        print(f"⚠️ No se pudo precargar las bases: {exc}")

KITCHEN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;600;700&display=swap');

:root {
    --cream: #faf6f0;
    --wood: #8b5e3c;
    --terracotta: #c45c26;
    --tomato: #d94f30;
    --herb: #5a8f5a;
    --butter: #f4d58d;
    --chalk: #2d3436;
}

.gradio-container {
    background: var(--cream) !important;
    background-image:
        linear-gradient(90deg, rgba(200,200,200,0.08) 1px, transparent 1px),
        linear-gradient(rgba(200,200,200,0.08) 1px, transparent 1px) !important;
    background-size: 24px 24px !important;
    font-family: 'Nunito', sans-serif !important;
}

.kitchen-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
    background: linear-gradient(135deg, #3d2914 0%, #5c3d1e 100%);
    border-radius: 16px;
    border: 4px solid var(--wood);
    box-shadow: 0 6px 20px rgba(61, 41, 20, 0.25);
    margin-bottom: 1rem;
}

.kitchen-header h1 {
    font-family: 'Pacifico', cursive !important;
    color: var(--butter) !important;
    font-size: 2.4rem !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.kitchen-header p {
    color: #f5e6d3 !important;
    margin: 0.4rem 0 0.8rem !important;
    font-size: 1rem !important;
}

.kitchen-garnish {
    text-align: center;
    color: var(--wood);
    font-size: 1.1rem;
    padding: 0.5rem;
    opacity: 0.85;
}

/* Recipe card chatbot */
#kitchen-chat {
    border: 3px solid var(--wood) !important;
    border-radius: 16px !important;
    background: #fffef9 !important;
    box-shadow: 0 8px 24px rgba(139, 94, 60, 0.15) !important;
}

#kitchen-chat .message.user {
    background: linear-gradient(135deg, #4a4a4a 0%, #2d3436 100%) !important;
    color: #f8f8f8 !important;
    border-radius: 12px 12px 4px 12px !important;
}

#kitchen-chat .message.bot {
    background: linear-gradient(180deg, #fffef9 0%, #f9f3e8 100%) !important;
    border: 1px solid #e8dcc8 !important;
    border-radius: 12px 12px 12px 4px !important;
    color: var(--chalk) !important;
}

/* Sticky-note input */
#kitchen-input textarea {
    background: #fff9c4 !important;
    border: 2px dashed var(--butter) !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    box-shadow: 2px 3px 8px rgba(0,0,0,0.08) !important;
}

#kitchen-input textarea::placeholder {
    color: #9a8b6e !important;
}

/* Cook button */
#cook-btn {
    background: linear-gradient(180deg, var(--tomato) 0%, var(--terracotta) 100%) !important;
    border: 2px solid #a63d1f !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(196, 92, 38, 0.4) !important;
    transition: transform 0.15s ease !important;
}

#cook-btn:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 16px rgba(196, 92, 38, 0.5) !important;
}

/* Example chips */
.examples button {
    background: #fff !important;
    border: 2px solid var(--herb) !important;
    color: var(--herb) !important;
    border-radius: 20px !important;
    font-size: 0.9rem !important;
}

.examples button:hover {
    background: var(--herb) !important;
    color: white !important;
}
"""

EXAMPLES = [
    "Tengo pollo, ¿qué puedo cocinar?",
    "Tengo frutas y hace mucho calor, ¿qué preparo?",
    "Soy atleta, necesito una comida alta en proteína",
    "¿Podés adaptar esta receta para 10 porciones en vez de 5?",
    "Necesito una receta rápida, ¡tengo hambre!",
    "Tengo tomate, queso y pan",
    "Solo tengo huevo y papa",
]

_session_threads: dict[str, str] = {}


def _get_thread_id(request: gr.Request | None) -> str:
    """Map each Gradio session to a stable LangGraph thread_id."""
    if request is None:
        return str(uuid.uuid4())

    session_hash = getattr(request, "session_hash", None)
    if session_hash is None:
        return str(uuid.uuid4())

    if session_hash not in _session_threads:
        _session_threads[session_hash] = str(uuid.uuid4())
    return _session_threads[session_hash]


async def stream_response(
    message: str,
    history: list[dict],
    request: gr.Request,
):
    """Stream a strict, database-grounded response into the chat UI."""
    if not message or not message.strip():
        yield history
        return

    thread_id = _get_thread_id(request)

    history = history + [{"role": "user", "content": message}]
    history = history + [{"role": "assistant", "content": ""}]
    yield history

    try:
        async for partial in stream_query(message, thread_id):
            history[-1]["content"] = partial
            yield history

    except FileNotFoundError as exc:
        history[-1]["content"] = f"⚠️ {exc}"
        yield history
    except Exception as exc:
        history[-1]["content"] = (
            f"⚠️ Ocurrió un error al procesar tu consulta: {exc}"
        )
        yield history


def create_app() -> gr.Blocks:
    """Build the kitchen-themed Gradio interface."""
    kitchen_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.stone,
        font=gr.themes.GoogleFont("Nunito"),
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_title_text_weight="600",
    )

    assets_dir = PROJECT_ROOT / "assets"
    user_avatar = str(assets_dir / "user.png") if (assets_dir / "user.png").exists() else None
    chef_avatar = str(assets_dir / "logo.png") if (assets_dir / "logo.png").exists() else None

    with gr.Blocks(title="Qué Cocinar IA") as demo:
        gr.HTML(
            """
            <div class="kitchen-header">
                <h1>🍳 Qué Cocinar IA</h1>
                <p>Tu asistente inteligente de cocina — recetas, sustituciones y más</p>
            </div>
            """
        )

        chatbot = gr.Chatbot(
            elem_id="kitchen-chat",
            height=520,
            avatar_images=(user_avatar, chef_avatar),
            show_label=False,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                elem_id="kitchen-input",
                placeholder="🍴 Contame qué ingredientes tenés o qué querés cocinar...",
                scale=5,
                show_label=False,
                container=False,
            )
            cook_btn = gr.Button("👨‍🍳 Cocinar", elem_id="cook-btn", scale=1, variant="primary")

        gr.Examples(
            examples=EXAMPLES,
            inputs=msg_input,
            label="🥕 Ideas para empezar",
        )

        gr.HTML(
            '<div class="kitchen-garnish">🥕 🍅 🧄 🌿 Hecho con amor en la cocina</div>'
        )

        async def respond(message, history, request: gr.Request):
            async for updated in stream_response(message, history, request):
                yield updated, ""

        cook_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )

    return demo, kitchen_theme


if __name__ == "__main__":
    _warmup_backend()
    demo, theme = create_app()
    demo.queue().launch(theme=theme, css=KITCHEN_CSS)
