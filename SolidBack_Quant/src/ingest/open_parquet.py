import pandas as pd
from pathlib import Path
import polars as pl
import pyarrow
def abrir_parquet_data_polar(nombre_archivo):
    """
    Sube un nivel, entra a la carpeta data y devuelve un LazyFrame de Polars.
    """

    ruta = Path("..") / "data" / nombre_archivo

    print(f"Cargando: {ruta}")

    return pl.scan_parquet(ruta)

#####################################################################################
from pathlib import Path
import pandas as pd

def abrir_parquet_data_pandas(nombre_archivo):
    """
    Sube un nivel, entra a la carpeta 'data' y abre el archivo especificado.
    """
    # Construye la ruta: sube un nivel (..) -> entra a data -> busca el nombre
    ruta = Path("..") / "data" / nombre_archivo
    
    # Lee y retorna el DataFrame
    print(f"Cargando exitosamente: {nombre_archivo}")
    return pd.read_parquet(ruta)
