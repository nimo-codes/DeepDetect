import gradio as gr
import inference_2 as inference


title="DEEPDETECT"
description="Deepfake detection"
            
           
video_inf = gr.Interface(inference.df_video_pred,
                    gr.Video(),
                    "text",
                    cache_examples = False
                    )


image_inf = gr.Interface(inference.df_img_pred,
                    gr.Image(),
                    "text",
                    cache_examples=False
                    )


app = gr.TabbedInterface(interface_list= [image_inf, video_inf], 
                         tab_names = ['Image inference', 'Video inference'])

if __name__ == '__main__':
    app.launch(share = True)