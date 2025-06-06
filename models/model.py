from llama_cpp import Llama

# Chemin vers le modèle .gguf
model_path = "/home/a798673/TestAI/models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"

# Initialisation du modèle (attention à la RAM dispo !)
def init_model():
    return Llama(
        model_path=model_path,
        n_ctx=2048,             # taille du contexte
        n_threads=8,            # adapte selon ton CPU
        n_gpu_layers=0          # 0 = tout sur CPU, >0 = couches sur GPU si disponible
    )

def ask(llm,prompt)-> str:
    """
    Envoie un prompt au modèle et retourne la réponse.
    """
    output = llm(prompt, max_tokens=200)
    return output["choices"][0]["text"].strip()