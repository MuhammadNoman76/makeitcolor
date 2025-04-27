import gradio as gr
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import uuid
from gradio_imageslider import ImageSlider

img_colorization = pipeline(Tasks.image_colorization, model='./makeitcolor')
img_path = 'input.png'

def color(image):
    output = img_colorization(image[...,::-1])
    result = output[OutputKeys.OUTPUT_IMG].astype(np.uint8)
    unique_imgfilename = str(uuid.uuid4()) + '.png'
    cv2.imwrite(unique_imgfilename, result)
    print('infer finished!')
    return (image, unique_imgfilename)
    

title = "Makeitcolor - Make Any Black & White image into color"
description = "upload old photo, dual decoder image colorization"
examples = [['examples/image1.jpg' , 'examples/image2.jpg' , 'examples/image3.jpg'],]

demo = gr.Interface(fn=color,inputs="image",outputs=ImageSlider(position=0.5,label='Colored image with slider-view'),examples=examples,title=title,description=description)

if __name__ == "__main__":
    demo.launch(share=False)