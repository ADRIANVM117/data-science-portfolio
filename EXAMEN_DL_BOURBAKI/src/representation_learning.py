from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


def build_autoencoder(input_dim=53, latent_dim=8):
    """
    Construye el autoencoder utilizado para aprender una representación
    latente de las trayectorias intradía.

    Arquitectura:
        input_dim -> 32 -> 16 -> latent_dim -> 16 -> 32 -> input_dim

    Parameters
    ----------
    input_dim : int
        Número de retornos intradía de entrada.

    latent_dim : int
        Dimensión del espacio latente.

    Returns
    -------
    autoencoder : keras.Model
        Modelo completo para reconstrucción.

    encoder : keras.Model
        Encoder que transforma la trayectoria original
        en su representación latente.
    """

    inputs = Input(
        shape=(input_dim,),
        name="returns_input"
    )

    # Encoder
    x = Dense(32, activation="relu")(inputs)
    x = Dense(16, activation="relu")(x)

    latent = Dense(
        latent_dim,
        activation="linear",
        name="latent"
    )(x)

    # Decoder
    x = Dense(16, activation="relu")(latent)
    x = Dense(32, activation="relu")(x)

    outputs = Dense(
        input_dim,
        activation="linear",
        name="reconstruction"
    )(x)

    autoencoder = Model(
        inputs=inputs,
        outputs=outputs,
        name="intraday_autoencoder"
    )

    encoder = Model(
        inputs=inputs,
        outputs=latent,
        name="intraday_encoder"
    )

    return autoencoder, encoder