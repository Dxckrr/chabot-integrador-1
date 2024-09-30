import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from googletrans import Translator

# Inicializar el traductor
translator = Translator()

# Cargar el modelo y el tokenizador de BioGPT
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

def get_chatbot_response(sentence_original):
    # Traducir la entrada al inglés
    sentence_en = translator.translate(sentence_original, src='es', dest='en').text
    inputs = tokenizer(sentence_en, return_tensors="pt")

    # Semilla que ayuda a la creación de la respuesta
    set_seed(42)

    # Generar la salida en inglés
    with torch.no_grad():
        beam_output = model.generate(**inputs,
                                     min_length=50,
                                     max_length=100,
                                     num_beams=5,
                                     early_stopping=True
                                     )

    # Decodificar la salida en inglés
    generated_text_english = tokenizer.decode(beam_output[0], skip_special_tokens=True)

    # Traducir la respuesta al español
    translated_text = translator.translate(generated_text_english, src='en', dest='es').text

    return translated_text
