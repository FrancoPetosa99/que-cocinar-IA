"""Estilos del frontend de Qué Cocinar IA."""

KITCHEN_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;600;700;800&display=swap');

/* ============================== */
/* Paletas de Colores Dinámicas   */
/* ============================== */
:root {
    --dark-amethyst: #10002bff;
    --dark-amethyst-2: #240046ff;
    --indigo-ink: #3c096cff;
    --indigo-velvet: #5a189aff;
    --royal-violet: #7b2cbfff;
    --lavender-purple: #9d4eddff;
    --mauve-magic: #c77dffff;
    --mauve: #e0aaffff;

    --bg-main: var(--mauve);
    --bg-panel: #ffffff;
    --text-main: var(--dark-amethyst);
    --text-muted: var(--indigo-ink);
    --border-color: var(--mauve-magic);
    --accent-primary: var(--royal-violet);
    --accent-hover: var(--indigo-velvet);
}

.dark {
    --dark-amethyst: #10002bff;
    --dark-amethyst-2: #240046ff;
    --indigo-ink: #3c096cff;
    --indigo-velvet: #5a189aff;
    --royal-violet: #7b2cbfff;
    --lavender-purple: #9d4eddff;
    --mauve-magic: #c77dffff;
    --mauve: #e0aaffff;

    --bg-main: var(--dark-amethyst);
    --bg-panel: var(--dark-amethyst-2);
    --text-main: #f8f9fbff;
    --text-muted: var(--mauve);
    --border-color: var(--indigo-velvet);
    --accent-primary: var(--lavender-purple);
    --accent-hover: var(--mauve-magic);
}

/* ============================== */
/* Estilos Generales              */
/* ============================== */
body, .gradio-container {
    background-color: var(--bg-main) !important;
    background-image: none !important;
    color: var(--text-main) !important;
    font-family: 'Nunito', sans-serif !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* ============================== */
/* Header + Barra de control      */
/* ============================== */
.kitchen-header-wrap {
    position: relative;
    margin-bottom: 1rem;
}

.kitchen-header {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
    background: var(--bg-panel);
    border-radius: 12px;
    border: 2px solid var(--border-color);
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.kitchen-header h1 {
    font-family: 'Pacifico', cursive !important;
    color: var(--accent-primary) !important;
    font-size: 2.4rem !important;
    margin: 0 !important;
}

.kitchen-header p {
    color: var(--text-muted) !important;
    margin: 0.4rem 0 0.8rem !important;
    font-size: 1rem !important;
}

/* ------------------------------------------------------------------ */
/* Barra de controles: posicionada arriba a la derecha sobre el header */
/* ------------------------------------------------------------------ */
#top-controls-row {
    position: absolute !important;
    top: 0.85rem;
    right: 0.85rem;
    z-index: 20;
    display: flex !important;
    flex-direction: row !important;
    width: auto !important;
    max-width: max-content !important;
    gap: 0.6rem !important;
    align-items: center !important;
    justify-content: flex-end !important;
}

/* Forzar a las columnas contenedoras de Gradio a no aplastar los botones */
#top-controls-row > * {
    width: auto !important;
    min-width: max-content !important;
    flex: 0 0 auto !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
}

/* ------------------------------------------------------------------ */
/* Botones superiores — diseño moderno tipo píldora horizontal       */
/* ------------------------------------------------------------------ */
#sidebar-toggle-btn,
#theme-toggle-btn {
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.5rem !important;

    /* Geometría estricta anti-deformación */
    height: 2.2rem !important;
    width: auto !important;
    min-width: max-content !important;
    padding: 0 1rem !important;
    border-radius: 50px !important;

    /* Estilos base (Claro) */
    background-color: var(--accent-primary)
    color: #ffffff !important;
    border: 1.5px solid var(--border-color) !important;

    /* Tipografía */
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    white-space: nowrap !important;

    box-shadow: 0 2px 5px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
}

/* Evitar que Gradio altere o colapse el texto interno del botón */
#sidebar-toggle-btn span,
#theme-toggle-btn span {
    display: inline-block !important;
    white-space: nowrap !important;
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}

/* Íconos renderizados vía ::before */
#sidebar-toggle-btn::before,
#theme-toggle-btn::before {
    content: '';
    display: inline-block;
    width: 1rem;
    height: 1rem;
    flex-shrink: 0;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    transition: background-image 0.18s ease;
}

/* Ícono menú (light) */
#sidebar-toggle-btn::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%237b2cbf' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='2'/%3E%3Cline x1='9' y1='4' x2='9' y2='20'/%3E%3C/svg%3E");
}

/* Ícono tema (light) */
#theme-toggle-btn::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%237b2cbf' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 3a9 9 0 0 0 0 18z' fill='%237b2cbf' stroke='none'/%3E%3C/svg%3E");
}

/* ------------------------------------------------------------------ */
/* Ajustes de Contraste Premium para el Modo Oscuro                   */
/* ------------------------------------------------------------------ */
.dark #sidebar-toggle-btn,
.dark #theme-toggle-btn {
    background-color: rgba(255, 255, 255, 0.08) !important; /* Esmerilado translúcido */
    color: #ffffff !important; /* Texto blanco puro */
    border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(4px) !important;
}

.dark #sidebar-toggle-btn::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='2'/%3E%3Cline x1='9' y1='4' x2='9' y2='20'/%3E%3C/svg%3E");
}

.dark #theme-toggle-btn::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 3a9 9 0 0 0 0 18z' fill='%23ffffff' stroke='none'/%3E%3C/svg%3E");
}

/* --- Estados Hover --- */
#sidebar-toggle-btn:hover,
#theme-toggle-btn:hover {
    background-color: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.18) !important;
    transform: translateY(-1px);
}

#sidebar-toggle-btn:hover::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='4' width='18' height='16' rx='2'/%3E%3Cline x1='9' y1='4' x2='9' y2='20'/%3E%3C/svg%3E") !important;
}

#theme-toggle-btn:hover::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 3a9 9 0 0 0 0 18z' fill='%23ffffff' stroke='none'/%3E%3C/svg%3E") !important;
}

/* --- Estados Active --- */
#sidebar-toggle-btn:active,
#theme-toggle-btn:active {
    transform: translateY(0);
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
}

/* ============================== */
/* Sidebar & Scrollbar            */
/* ============================== */
#cuisine-sidebar-col {
    max-height: 75vh !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-primary) var(--bg-panel);
    padding-right: 8px;
    flex: 0 0 320px !important;
    max-width: 320px !important;
}

#cuisine-sidebar-col::-webkit-scrollbar {
    width: 6px;
}
#cuisine-sidebar-col::-webkit-scrollbar-track {
    background: var(--bg-panel);
    border-radius: 4px;
}
#cuisine-sidebar-col::-webkit-scrollbar-thumb {
    background-color: var(--accent-primary);
    border-radius: 4px;
}

#cuisine-sidebar {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
}

#cuisine-sidebar .label-wrap {
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

.cuisine-subcat-accordion {
    margin: 0.25rem 0 0.5rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    background: var(--bg-main) !important;
}

.cuisine-subcat-accordion .label-wrap {
    color: var(--accent-primary) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

.cuisine-subcat-btn {
    background: transparent !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.4rem 0.6rem !important;
    margin: 0.15rem 0 !important;
    transition: all 0.2s ease !important;
}

.cuisine-subcat-btn:hover {
    background: var(--accent-primary) !important;
    color: #ffffff !important;
    border-color: var(--accent-primary) !important;
}

.cuisine-recipe-gallery {
    background: transparent !important;
    border: none !important;
}

.cuisine-recipe-gallery .grid-wrap {
    background: transparent !important;
}

.cuisine-recipe-gallery [data-testid="grid"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 1rem !important;
    grid-template-columns: none !important;
}

.cuisine-recipe-gallery .thumbnail-item,
.cuisine-recipe-gallery .grid-container .thumbnail-item {
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background: var(--bg-panel) !important;
    transition: all 0.2s ease !important;
    width: 90% !important;
    aspect-ratio: auto !important;
    overflow: hidden;
}

.cuisine-recipe-gallery .thumbnail-item img,
.cuisine-recipe-gallery .grid-container .thumbnail-item img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

.cuisine-recipe-gallery .thumbnail-item:hover,
.cuisine-recipe-gallery .grid-container .thumbnail-item:hover {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px var(--accent-primary) !important;
    transform: translateY(-2px);
}

.cuisine-recipe-gallery .caption-label {
    color: var(--text-main) !important;
    background: var(--bg-panel) !important;
    font-size: 0.75rem !important;
}

/* ============================== */
/* Chatbot UI                     */
/* ============================== */
#kitchen-chat {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--bg-panel) !important;
}

#kitchen-chat .message.user {
    background: var(--accent-primary) !important;
    color: #ffffff !important;
    border-radius: 14px 14px 4px 14px !important;
}

#kitchen-chat .message.bot {
    background: var(--bg-main) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-main) !important;
    border-radius: 14px 14px 14px 4px !important;
}

#kitchen-chat p, #kitchen-chat li, #kitchen-chat span {
    color: inherit !important;
}

/* ============================== */
/* Inputs y Botones Generales     */
/* ============================== */
#kitchen-input textarea {
    background: var(--bg-panel) !important;
    color: var(--text-main) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
}

#kitchen-input textarea:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
}

#cook-btn {
    background: var(--accent-primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: background 0.2s ease !important;
}

#cook-btn:hover {
    background: var(--accent-hover) !important;
}

/* ============================== */
/* Ejemplos (gr.Examples)         */
/* Elimina completamente el       */
/* naranja en el tema claro.      */
/* ============================== */

/* Contenedor y encabezado del bloque de ejemplos */
.gradio-container .examples,
.gradio-container [class*="examples"] {
    background: transparent !important;
}

/* Tabla de ejemplos */
.gradio-container .gr-samples-table,
.gradio-container table.gr-samples-table,
.gradio-container table[class*="samples"],
.gradio-container .label {
    color: var(--text-main) !important;
    background: transparent !important;
}

/* Filas y celdas individuales */
.gradio-container .gr-sample-row,
.gradio-container .gr-sample-textbox,
.gradio-container td.svelte-1oa6fve,
.gradio-container tr.svelte-1oa6fve td,
.gradio-container .samples-holder td,
.gradio-container table td {
    background: var(--bg-panel) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease !important;
}

/* Hover de filas: acento violeta, sin ningún naranja */
.gradio-container .gr-sample-row:hover,
.gradio-container .gr-sample-textbox:hover,
.gradio-container .gr-sample-row:hover *,
.gradio-container .gr-sample-textbox:hover *,
.gradio-container td.svelte-1oa6fve:hover,
.gradio-container tr.svelte-1oa6fve:hover td,
.gradio-container .samples-holder tr:hover td,
.gradio-container table tr:hover td {
    background: var(--accent-primary) !important;
    color: #ffffff !important;
    border-color: var(--accent-primary) !important;
}

/* Cancelar cualquier color naranja que Gradio pueda inyectar
   a través de sus variables CSS propias */
.gradio-container {
    --color-accent: var(--accent-primary) !important;
    --color-accent-soft: var(--mauve) !important;
    --button-primary-background-fill: var(--accent-primary) !important;
    --button-primary-background-fill-hover: var(--accent-hover) !important;
    --button-secondary-border-color: var(--border-color) !important;
    --button-secondary-border-color-hover: var(--accent-primary) !important;
    --button-secondary-text-color: var(--text-main) !important;
    --button-secondary-text-color-hover: #ffffff !important;
    --button-secondary-background-fill-hover: var(--accent-primary) !important;
    --examples-bg: transparent !important;
    --table-row-focus: var(--accent-primary) !important;
}

/* ============================== */
/* Responsive: sidebar + chatbot  */
/* ============================== */
@media (max-width: 900px) {
    #cuisine-sidebar-col {
        flex: 1 1 100% !important;
        max-width: 100% !important;
        max-height: 50vh !important;
    }

    .cuisine-recipe-gallery .thumbnail-item,
    .cuisine-recipe-gallery .grid-container .thumbnail-item {
        width: 100% !important;
    }
}
"""