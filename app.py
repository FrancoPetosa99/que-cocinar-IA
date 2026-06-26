import os
import gradio as gr
from google import genai
from dotenv import load_dotenv
from system_prompt import agent_role

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm_model = os.getenv("LLM_MODEL")
llm_client = genai.Client(api_key=api_key)


def responder(user_message, chat):

    if not es_consulta_cocina(user_message):
        chat.append({"role": "user"     , "content": user_message})
        chat.append({"role": "assistant", "content": "🍳 Lo siento, solo puedo ayudar con recetas y temas relacionados con la cocina."})
        return chat, ""

    response = llm_client.models.generate_content(
        model=llm_model,
        contents=f"""

            {agent_role}

            CONSULTA DEL USUARIO: 
            {user_message}
        
        """
    )

    answer = response.text

    chat.append({"role": "user",      "content": user_message})
    chat.append({"role": "assistant", "content": answer      })

    return chat, ""

def es_consulta_cocina(user_message):
    prompt = f"""
    Responde únicamente SI o NO.

    ¿La siguiente consulta está relacionada con cocina,
    recetas, ingredientes o alimentación?

    Consulta: {user_message}
    """

    response = llm_client.models.generate_content(
        model=llm_model,
        contents=prompt
    )

    return "SI" in response.text.upper()

with gr.Blocks(title="Qué Cocinar IA") as demo:

    gr.Markdown("# 🍳 Qué Cocinar IA")
    gr.Markdown("Asistente de recetas basado en LLM")

    with gr.Row():

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=600,
                avatar_images=("assets/user.png", "assets/logo.png")
            )

    ingredientes = gr.Textbox(
        placeholder="pollo, arroz, cebolla..."
    )

    gr.Examples(
        examples=[
            "Tengo pollo y arroz",
            "Quiero una receta vegetariana",
            "Tengo tomate, queso y pan",
            "Solo tengo huevo y papa",
            "Arroz, atún y mayonesa"
        ],
        inputs=ingredientes
    )

    enviar = gr.Button("Enviar", variant="primary")

    enviar.click(
        fn=responder,
        inputs=[ingredientes, chatbot],
        outputs=[chatbot, ingredientes]
    )

    ingredientes.submit(
        fn=responder,
        inputs=[ingredientes, chatbot],
        outputs=[chatbot, ingredientes]
    )

demo.launch()