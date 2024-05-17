import gradio as gr

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("hi") as hi:
            pass
        with gr.Tab("hello") as hello:
            pass
        
    output = gr.Textbox()
    hi.select(lambda :"hi", None, output)
    hello.select(lambda :"hello", None, output) 
    
demo.launch()