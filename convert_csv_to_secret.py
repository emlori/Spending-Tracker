import base64
import pandas as pd

def convert_csv_to_base64(csv_file_path):
    # Lire le fichier CSV
    with open(csv_file_path, 'rb') as file:
        csv_content = file.read()
    
    # Convertir en base64
    base64_content = base64.b64encode(csv_content).decode('utf-8')
    
    # Créer le contenu pour le fichier secrets.toml
    toml_content = f'tricount_data = """{base64_content}"""'
    
    # Écrire dans un fichier temporaire
    with open('secrets_temp.toml', 'w') as file:
        file.write(toml_content)
    
    print("Le fichier secrets_temp.toml a été créé avec succès.")
    print("Copiez son contenu dans les secrets de votre application Streamlit.")

if __name__ == "__main__":
    convert_csv_to_base64("Tricount_Switzerland.csv") 