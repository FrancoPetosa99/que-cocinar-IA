"""Kitchen-themed Gradio chat interface with token streaming."""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.pipeline import stream_query
from styles import KITCHEN_CSS
from cuisine_tree import build_cuisine_tree, label_cuisine_es, resolve_recipes_csv_path


def _warmup_backend() -> None:
    """Load SQLite, Chroma, and optionally the local HF LLM at startup."""
    try:
        from backend.config import (
            CHROMA_DIR,
            HF_BACKEND,
            LLM_PROVIDER,
            SQLITE_PATH,
            preload_local_llm,
        )
        from backend.database import get_vectorstore, reset_vectorstore
        from backend.recipe_db import get_connection

        reset_vectorstore()
        get_connection()
        get_vectorstore()
        print(f"✓ SQLite lista ({SQLITE_PATH})")
        print(f"✓ Chroma lista ({CHROMA_DIR})")

        if LLM_PROVIDER == "huggingface" and HF_BACKEND == "local":
            print("Cargando modelo Hugging Face local (puede tardar unos minutos)...")
            preload_local_llm()
            print("✓ Modelo local listo")
    except Exception as exc:
        print(f"⚠️ No se pudo precargar el backend: {exc}")


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
    cuisine_tree = build_cuisine_tree(resolve_recipes_csv_path())
    english_roots = [
        root
        for root in cuisine_tree
        if root in {"Desserts", "Cakes", "Cookies", "Bread", "Salad", "Seafood"}
        or (root == label_cuisine_es(root) and re.fullmatch(r"[A-Za-z ,&':]+", root))
    ]
    sample_roots = list(cuisine_tree.keys())[:5]
    if cuisine_tree:
        print(
            f"✓ Sidebar: {len(cuisine_tree)} categorías "
            f"(ej. {', '.join(sample_roots)})"
        )
        if english_roots:
            print(
                "⚠️ Sidebar: categorías sin traducir detectadas: "
                f"{', '.join(english_roots[:5])}"
            )
    else:
        print("⚠️ Sidebar: no se cargaron categorías (revisá data/recipes_spanish.csv)")

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
        # ------------------------------------------------------------
        # Header + barra de controles (tema / mostrar-ocultar sidebar).
        # Los botones se posicionan arriba a la derecha (vía CSS, ver
        # #top-controls-row en styles.py) en vez de vivir dentro del
        # sidebar, donde antes lo ensanchaban y lo hacían incómodo.
        # ------------------------------------------------------------
        with gr.Column(elem_classes=["kitchen-header-wrap"]):
            gr.HTML(
                """
                <div class="kitchen-header">
                    <h1>🍳 Qué Cocinar IA</h1>
                    <p>Tu asistente inteligente de cocina — recetas, sustituciones y más</p>
                </div>
                """
            )
            with gr.Row(elem_id="top-controls-row"):
                sidebar_toggle_btn = gr.Button(
                    "Menú",
                    elem_id="sidebar-toggle-btn",
                    elem_classes=["icon-btn"],
                    size="sm",
                    icon=None,
                )
                theme_toggle_btn = gr.Button(
                    "Tema",
                    elem_id="theme-toggle-btn",
                    elem_classes=["icon-btn"],
                    size="sm",
                    icon=None,
                )

        with gr.Row():
            # ------------------------------------------------------------
            # Sidebar de categorías (basado en cuisine_path), de 3 niveles:
            #   Categoría (Accordion) -> Subcategoría (Accordion anidado)
            #   -> Recetas (Gallery con imagen + nombre).
            # Solo UI: al elegir una receta, dispara directamente la consulta
            # al chat sin mostrar ningún detalle previo.
            # ------------------------------------------------------------
            with gr.Column(scale=1, min_width=220, elem_id="cuisine-sidebar-col") as sidebar_col:
                sidebar_visible = gr.State(True)
                with gr.Column(elem_id="cuisine-sidebar"):
                    gr.Markdown(
                        "**🍽️ Categorías de recetas**",
                        elem_classes=["cuisine-sidebar-title"],
                    )

                    subcat_fallback_buttons: list[tuple[gr.Button, str, str]] = []
                    recipe_galleries: list[tuple[gr.Gallery, list[str]]] = []

                    if cuisine_tree:
                        for root_category, subcategories in cuisine_tree.items():
                            root_label = label_cuisine_es(root_category)
                            with gr.Accordion(label=root_label, open=False):
                                if subcategories:
                                    for subcat, recipes in subcategories.items():
                                        with gr.Accordion(
                                            label=label_cuisine_es(subcat),
                                            open=False,
                                            elem_classes=["cuisine-subcat-accordion"],
                                        ):
                                            if recipes:
                                                gallery_items = [
                                                    (r.image_url, r.name)
                                                    for r in recipes
                                                    if r.image_url
                                                ]
                                                recipe_names = [
                                                    r.name for r in recipes if r.image_url
                                                ]
                                                if gallery_items:
                                                    gallery = gr.Gallery(
                                                        value=gallery_items,
                                                        label=None,
                                                        show_label=False,
                                                        columns=1,
                                                        height="auto",
                                                        object_fit="contain",
                                                        elem_classes=["cuisine-recipe-gallery"],
                                                        allow_preview=False,
                                                    )
                                                    recipe_galleries.append((gallery, recipe_names))
                                                else:
                                                    gr.Markdown(
                                                        f"_Sin imágenes disponibles para {subcat}_",
                                                        elem_classes=["cuisine-subcat-btn"],
                                                    )
                                            else:
                                                gr.Markdown(
                                                    f"_Sin recetas cargadas para {subcat}_",
                                                    elem_classes=["cuisine-subcat-btn"],
                                                )
                                else:
                                    btn = gr.Button(
                                        f"Ver recetas de {root_category}",
                                        elem_classes=["cuisine-subcat-btn"],
                                        size="sm",
                                    )
                                    subcat_fallback_buttons.append((btn, root_category, root_category))
                    else:
                        gr.Markdown(
                            "_No se encontraron categorías (data/recipes_spanish.csv no disponible)._"
                        )

            with gr.Column(scale=4):
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

        # Toggle de visibilidad del sidebar (solo UI)
        def _toggle_sidebar(is_visible: bool):
            new_state = not is_visible
            label = "Menú" if new_state else "Menú"
            return gr.Column(visible=new_state), new_state, gr.Button(label)

        sidebar_toggle_btn.click(
            _toggle_sidebar,
            inputs=[sidebar_visible],
            outputs=[sidebar_col, sidebar_visible, sidebar_toggle_btn],
        )

        theme_toggle_btn.click(
            None,
            None,
            None,
            js="""
            function() {
                document.body.classList.toggle('dark');
            }
            """
        )

        # Botones fallback (categoría sin subcategorías): dispara directamente
        # la consulta al chat sin pasar por el textbox.
        for btn, root_category, subcat in subcat_fallback_buttons:
            query_text = f"Quiero una receta de {subcat} ({root_category})"

            def _make_respond_from_query(_query=query_text):
                async def _respond_direct(history, request: gr.Request):
                    async for updated in stream_response(_query, history, request):
                        yield updated, ""
                return _respond_direct

            btn.click(
                _make_respond_from_query(),
                inputs=[chatbot],
                outputs=[chatbot, msg_input],
            )

        # Galería de recetas (nivel 3): al seleccionar una imagen, dispara
        # directamente la consulta al chat sin mostrar ningún detalle previo.
        for gallery, recipe_names in recipe_galleries:

            def _make_gallery_respond(_names=recipe_names):
                async def _gallery_respond(evt: gr.SelectData, history, request: gr.Request):
                    index = evt.index if evt is not None else None
                    if index is None or index >= len(_names):
                        yield history, ""
                        return
                    query = f"Quiero la receta de {_names[index]}"
                    async for updated in stream_response(query, history, request):
                        yield updated, ""
                return _gallery_respond

            gallery.select(
                _make_gallery_respond(),
                inputs=[chatbot],
                outputs=[chatbot, msg_input],
            )

    return demo, kitchen_theme


if __name__ == "__main__":
    _warmup_backend()
    demo, theme = create_app()
    demo.queue().launch(
        theme=theme,
        css=KITCHEN_CSS,
        show_api=False,
        favicon_path=str(PROJECT_ROOT / "assets" / "logo.png")
        if (PROJECT_ROOT / "assets" / "logo.png").exists()
        else None,
    )