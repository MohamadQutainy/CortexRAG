import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

from config import get_config
from rag.observability.logger import setup_logging
from rag.pipeline import build_generator

cfg = get_config()
setup_logging(
    log_level=cfg["observability"]["log_level"],
    log_file=cfg["observability"]["log_file"],
)


def _get_generator():
    
    if not hasattr(_get_generator, "_instance"):
        _get_generator._instance = build_generator()
    return _get_generator._instance


def format_context(context):

    result = "<h2 style='color: #c85a11;'>Retrieved Context</h2>\n\n"
    for index, doc in enumerate(context, start=1):
        source = doc.metadata.get("source", "unknown")
        result += "<details><summary style='color: #c85a11; cursor: pointer;'>"
        result += f"<b>[{index}]</b> {source}</summary>\n\n"
        result += doc.page_content + "\n\n</details>\n\n"
    return result


def chat(history):
    
    last_message = history[-1]["content"]
    if isinstance(last_message, tuple) or isinstance(last_message, list):
        last_message = last_message[0]
        
    prior_history = history[:-1]

    generator = _get_generator()
    answer, context = generator.answer(str(last_message), prior_history)

    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    

    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    company = cfg["prompts"]["company_name"]
    theme = gr.themes.Soft()

    with gr.Blocks(title=f"{company} Expert Assistant") as ui:
        gr.Markdown(f"# {company} Expert Assistant\nAsk me anything about {company}.")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    #type="messages",
                    #show_copy_button=True,
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder=f"Ask anything about {company}...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved Context",
                    value="Retrieved context will appear here.",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_markdown],
        )

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()
