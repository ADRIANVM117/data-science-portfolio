import json
import httpx

def obtener_token_activo():
    # URL oficial configurada para traer el mercado activo con mayor volumen
    url = "https://polymarket.com"
    
    # Headers para emular una petición de navegador estándar (Evita bloqueos)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=10)
        
        # Validar si el servidor devolvió un código de error (como 403 o 400)
        if response.status_code != 200:
            print(f"Error de Servidor: Código {response.status_code}. El endpoint rechazó la petición.")
            print(f"Respuesta cruda: {response.text[:200]}")
            return

        data = response.json()
        
        if data and isinstance(data, list):
            market = data[0]  # Tomamos el primer mercado de la lista
            clob_token_ids = market.get("clobTokenIds")
            title = market.get("title")
            
            if clob_token_ids:
                # El campo clobTokenIds de Gamma a veces viene serializado como string JSON
                tokens = json.loads(clob_token_ids) if isinstance(clob_token_ids, str) else clob_token_ids
                
                print(f"\n[✓] Mercado Encontrado con Éxito:")
                print(f"    Título: {title}")
                print(f"    Tokens Activos (YES/NO): {tokens}\n")
                
                print(f"Copia este ID en tu main.py: '{tokens[0]}'")
                return
                
        print("[-] No se encontraron mercados activos con volumen en esta respuesta.")
        
    except json.JSONDecodeError:
        print("[-] Error crítico: La respuesta del servidor no es un JSON válido.")
        print(f"    Contenido recibido del servidor:\n{response.text[:300]}")
    except Exception as e:
        print(f"[-] Error de conexión inesperado: {e}")

if __name__ == "__main__":
    obtener_token_activo()

