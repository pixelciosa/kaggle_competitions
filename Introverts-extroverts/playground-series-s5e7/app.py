import gradio as gr
import pandas as pd
import pickle
import numpy as np

# --- 1. Features & Model Setup ---
PARAMS_NAME = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
]

MODEL_PATH = "models/model.pkl"
with open(MODEL_PATH, 'rb') as handle:
    model = pickle.load(handle)

# --- 2. Updated Prediction Function ---
def predict(*args):
    # Create dictionary for DataFrame
    answer_dict = {PARAMS_NAME[i]: [args[i]] for i in range(len(PARAMS_NAME))}
    single_instance = pd.DataFrame.from_dict(answer_dict)

    # Ensure feature order
    features = PARAMS_NAME
    prediction = model.predict(single_instance[features])[0]

    # Map numerical prediction to Label and Image path
    label_map = {0: 'Introvert', 1: 'Extrovert'}
    image_map = {
        'Introvert': "images/Introverts.png",
        'Extrovert': "images/Extroverts.png"
    }

    result_label = label_map[prediction]
    result_image = image_map[result_label]

    # Return both values to update two components
    return result_label, result_image

# --- 3. Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center'>Introvert or Extrovert Predictor</h1>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Things about you:")
            Time_spent_Alone = gr.Slider(label="Time Spent Alone", minimum=0, maximum=11, step=1, value=1)
            Stage_fear = gr.Radio(label="Do you have stage fear?", choices=[True, False], value=True)
            Social_event_attendance = gr.Slider(label="Social event attendance", minimum=0, maximum=10, step=1, value=2)
            Going_outside = gr.Slider(label="Weekly social outings", minimum=0, maximum=7, step=1, value=4)
            Drained_after_socializing = gr.Radio(label="Feel drained after socializing?", choices=[True, False], value=True)
            Friends_circle_size = gr.Slider(label="Size of friends circle", minimum=0, maximum=15, step=1, value=2)
            Post_frequency = gr.Slider(label="Posts per week", minimum=0, maximum=11, step=1, value=1)
            
            predict_btn = gr.Button(value="Go Predict!", variant="primary")

        with gr.Column():
            gr.Markdown("## Prediction")
            label_output = gr.Label(label="You seem to be more of an ...")
            # This component will receive the image path dynamically
            image_output = gr.Image(label="Result Visualization", interactive=False)

    # Link the button to the function and define multiple outputs
    predict_btn.click(
        fn=predict,
        inputs=[
            Time_spent_Alone, Stage_fear, Social_event_attendance,
            Going_outside, Drained_after_socializing, Friends_circle_size, Post_frequency
        ],
        outputs=[label_output, image_output]
    )

    gr.Markdown(
        """
        <p style='text-align: center'>Model trained by Adriana Villalobos based on data from the  
            <a href='https://www.kaggle.com/competitions/playground-series-s5e7/overview' target='_blank'>Kaggle Competition</a>
        </p>
        """
    )

if __name__ == "__main__":
    demo.launch()