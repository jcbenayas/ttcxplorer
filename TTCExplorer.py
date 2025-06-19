import gradio as gr
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import traceback
from pathlib import Path
import re
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

COLOR_MAP = {
    "Receiver1": {"voltage": "hotpink", "status": "hotpink"},
    "Receiver2": {"voltage": "cadetblue", "status": "cadetblue"},
    "Receiver3": {"voltage": "orange", "status": "orange"},
    "status": {"voltage": "gold", "status": "gold"}
}

# Ajusta estos valores según tu instalación local
OPENWEBUI_URL = "https://tesmadslapp31"  # URL de tu servidor OpenWebUI
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImQ1ZDBlOWQ3LWIwNTgtNDBhZS1hODQ0LTUzZmU3YjA0ODZkZiJ9.dK51Qk5kydF_1TEz7lHXrydwCopwmOO7yTVD2NS3q3E"
MODEL_NAME = "ttcxplorer"  # Modelo a usar


# Construcción de los headers y sesión
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
session = requests.Session()
session.trust_env = False  # Ignora variables de entorno de proxy

# Variables globales para análisis y datos de anomalías
anomalies_global = None
series_global = None
anomalies_global_bloqueos = None
series_global_bloqueos= None
anomalies_global_moc = None
series_global_moc= None
anomalies_global_mlib = None
series_global_mlib= None

def generate_sql_query(question: str) -> str:
    """
    Convierte una pregunta en lenguaje natural en una consulta de Polars
    utilizando el endpoint de chat de OpenWebUI.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": question}
        ]
    }
    # Llamada al endpoint de chat completions
    resp = session.post(
        f"{OPENWEBUI_URL}/api/chat/completions", headers=HEADERS,
        json=payload, verify=False
    )
    resp.raise_for_status()
    data = resp.json()
    # Extraer la consulta generada
    query = data["choices"][0]["message"]["content"]
    return query.strip()

def execute_query(df: pl.DataFrame, query: str):
    try:
        ctx = pl.SQLContext()
        ctx.register("df", df)
        result = ctx.execute(query, eager=True)
    except Exception as e:
        return None, None, f"Error al ejecutar la consulta SQL: {e}"

    if isinstance(result, pl.DataFrame):
        pdf = result.to_pandas()
        if pdf.empty:
            return None, None, result
        if set(["name","time","value"]).issubset(pdf.columns):
            # Convert original 'time' column to timezone-naive datetimes and drop invalid
            pdf["time"] = pd.to_datetime(pdf["time"], utc=True, errors="coerce").dt.tz_convert(None)
            pdf = pdf.dropna(subset=["time"])
            # Pivot to wide form so that each distinct 'name' produces its own y-series
            wide_df = pdf.pivot(index="time", columns="name", values="value").reset_index()
            # Match width of detailed plots
            fig, ax = plt.subplots(figsize=(12, 4))
            for col in wide_df.columns:
                if col != "time":
                    ax.plot(wide_df["time"], wide_df[col], label=col)
            ax.set_title("Resultado de la consulta")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Valor")
            ax.legend()
            plt.tight_layout()
            fig_update   = gr.update(value=fig,   visible=(fig   is not None))
            table_update = gr.update(value=wide_df, visible=(wide_df is not None))
            return fig_update, table_update, f"Ejecución correcta de consulta SQL: {query}"
        else:
            table_update = gr.update(value=pdf, visible=(pdf is not None))
            return None, table_update, f"Ejecución correcta de consulta SQL: {query}"
    else:
        return None, None, result


def ask_polars(question: str, medidas):
    # Genera únicamente la consulta SQL sin ejecutar nada
    query = generate_sql_query(question)
    return query

# Nueva función para ejecutar una consulta Polars directamente
def run_polars_query(query: str, medidas):
    # Strip triple backtick fences if present in the query input
    sql = query
    if medidas is None or not getattr(medidas, "name", None):
        return None, None, "Por favor suba un archivo CSV o Parquet con las medidas."
    extension = Path(medidas.name).suffix.lower()
    if extension == ".csv":
        df_local = pl.read_csv(
            medidas.name,
            infer_schema_length=10000,
            schema_overrides={"value": pl.Float64},
            null_values=["", "NA", "null"],
            ignore_errors=True
        )
    elif extension == ".parquet":
        df_local = pl.read_parquet(medidas.name)
    else:
        return None, None, "Formato de archivo no soportado."
    fig, table, text = execute_query(df_local, sql)
    return fig, table, text

# Procesar Datos Caídas desde CSV o Parquet
def procesar_datos(medidas):
    global anomalies_global
    try:
        print(f"Archivo recibido: {medidas.name}")

        # Lectura dinámica del archivo medidas (CSV o Parquet)
        extension = Path(medidas.name).suffix.lower()

        if extension == ".csv":
            datos = pl.read_csv(
                medidas.name,
                infer_schema_length=10000,
                schema_overrides={"value": pl.Float64},
                null_values=["", "NA", "null"],
                ignore_errors=True
            )
        elif extension == ".parquet":
            datos = pl.read_parquet(medidas.name)
        else:
            return None, "Formato de archivo no soportado."

        print(f"Datos cargados: {datos.height} filas.")

        datos = datos.with_columns(
            pl.col('time').str.to_datetime(strict=False),
            pl.col('value').cast(pl.Float64, strict=False)
        ).drop_nulls('time')

        anomalies = []
        transceivers = datos.select(pl.col("name").str.split("_").list.first()).unique().to_series().to_list()

        for transceiver in transceivers:
            datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{transceiver}_"))
            if datos_filtrados.height != 0:
                status_col = f"{transceiver}_status"
                if status_col in datos_filtrados.get_column('name').to_list():
                    datos_status = datos_filtrados.filter(
                        (pl.col('name') == status_col) & (pl.col('value') == 1.0)
                    ).select('time')
                    datos_final = datos_filtrados.join(datos_status, on='time', how='inner')
                else:
                    continue

                filtered_columns = [name for name in datos_final.get_column('name').unique() if name.startswith(f"{transceiver}_voltageReceiver")]
                for c in filtered_columns:
                    values_col = datos_final.filter(pl.col('name') == c)['value']
                    mean_value = values_col.mean()
                    max_value = values_col.max()
                    percentage_difference = ((max_value - mean_value) / mean_value) * 100 if mean_value != 0 else 0

                    if percentage_difference >= 7.5:
                        receptor = c[-1]
                        anomalies.append({
                            'Transceptor': transceiver,
                            'Receptor': receptor,
                            'Diferencia': percentage_difference
                        })

        anomalies_global = pl.DataFrame(anomalies)
        if anomalies_global.is_empty():
            return None, "No se encontraron anomalías para las fechas seleccionadas."
        
        print(f"Procesamiento finalizado con {anomalies_global.height} anomalías.")
        return gr.update(value=anomalies_global.to_pandas(), visible=True), "Datos procesados exitosamente."

    except Exception as e:
        print("Error en procesar_datos:", e)
        anomalies_global = None
        return None, str(e)

#Procesar datos de bloqueo desde CSV
def procesar_datos_bloqueo(medidas):
    global anomalies_global_bloqueos
    try:
        extension = Path(medidas.name).suffix.lower()

        if extension == ".csv":
            datos = pl.read_csv(
                medidas.name,
                infer_schema_length=10000,
                schema_overrides={"value": pl.Float64},
                null_values=["", "NA", "null"],
                ignore_errors=True
            )
        elif extension == ".parquet":
            datos = pl.read_parquet(medidas.name)
        else:
            return None, "Formato de archivo no soportado."

        print(f"Datos cargados: {datos.height} filas.")

        datos = datos.with_columns([
            pl.col("time").str.to_datetime(strict=False),
            pl.col("value").cast(pl.Float64, strict=False)
        ]).drop_nulls("time")

        registros = []
        transceivers = datos.select(pl.col("name").str.split("_").list.first()).unique().to_series().to_list()
        for transceiver in transceivers:
            datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{transceiver}_"))

            if datos_filtrados.is_empty():
                continue

            pivot = datos_filtrados.pivot(index="time", values="value", on="name").sort("time")

            if pivot.is_empty():
                continue

            pivot = pivot.with_columns([
                (pl.col("time").cast(pl.Datetime("ns")) - pl.col("time").cast(pl.Datetime("ns")).shift(1))
                .cast(pl.Duration("ns")).cast(pl.Int64).alias("Tiempo")
            ])
            pivot = pivot.with_columns([
                (pl.col("Tiempo") / 1_000_000_000).cast(pl.Int64).alias("Tiempo")
            ])

            bloqueos = pivot.filter(pl.col("Tiempo") >= 180)

            if not bloqueos.is_empty():
                bloqueos = bloqueos.with_columns([
                    pl.lit(transceiver).alias("Transceptor")
                ])
                registros.append(bloqueos.select([
                    pl.col("time").alias("Timestamp"),
                    "Transceptor", "Tiempo"
                ]))

        if not registros:
            anomalies_global_bloqueos = pl.DataFrame([])
            return [], "No se encontraron anomalías para las fechas seleccionadas."

        anomalies = pl.concat(registros)
        anomalies_global_bloqueos = anomalies

        transceptores = anomalies.select("Transceptor").unique().get_column("Transceptor").to_list()
        print(f"Procesamiento finalizado con {len(transceptores)} transceptores únicos.")
        return gr.update(choices=transceptores, visible=True), gr.update(value="Datos procesados exitosamente.", visible=True)

    except Exception as e:
        print("Error en procesar_datos:", e)
        anomalies_global_bloqueos = None
        return [], f"Error al procesar los datos: {str(e)}"


#Procesar datos de microocupación desde CSV
def procesar_datos_moc(medidas):
    global anomalies_global_moc
    try:
        print(f"Archivo recibido: {medidas.name}")

        extension = Path(medidas.name).suffix.lower()
        if extension == ".csv":
            datos = pl.read_csv(
                medidas.name,
                infer_schema_length=10000,
                schema_overrides={"value": pl.Float64},
                null_values=["", "NA", "null"],
                ignore_errors=True
            )
        elif extension == ".parquet":
            datos = pl.read_parquet(medidas.name)
        else:
            return None, "Formato de archivo no soportado."

        datos = datos.with_columns([
            pl.col("time").str.to_datetime(strict=False),
            pl.col("value").cast(pl.Float64, strict=False)
        ]).drop_nulls("time")

        registros = []

        transceivers = datos.select(pl.col("name").str.split("_").list.first()).unique().to_series().to_list()
        for transceiver in transceivers:
            datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{transceiver}_status"))
            if datos_filtrados.is_empty():
                continue

            pivot = datos_filtrados.pivot(index="time", values="value", columns="name", aggregate_function="first").sort("time")

            status_col = f"{transceiver}_status"
            receiver_cols = [col for col in pivot.columns if col.startswith(f"{transceiver}_statusReceiver")]

            df = pivot.with_columns((pl.col(status_col) - pl.col(status_col).shift(1)).alias("change"))
            for r in receiver_cols:
                df = df.with_columns((pl.col(r) - pl.col(r).shift(1)).alias(f"change_{r}"))
            df = df.filter(pl.col("change") != 0.0)
            df = df.with_columns((pl.col("time").cast(pl.Datetime("ms")) - pl.col("time").cast(pl.Datetime("ms")).shift(1))
                .cast(pl.Duration("ms")).cast(pl.Int64).fill_null(0).alias("Tiempo"))
            moc = df.filter((pl.col("change") == 1.0) & pl.col("Tiempo").is_between(250, 2000))

            if not moc.is_empty():
                tipo_anomalia = []
                for row in moc.iter_rows(named=True):
                    cambios = [row.get(f"change_{r}", 0) for r in receiver_cols]
                    if all(c == 0 for c in cambios):
                        tipo_anomalia.append("bitcode/cascada")
                    else:
                        tipo = "desconocido"
                        for i, c in enumerate(cambios):
                            if c == 1:
                                tipo = f"receptor {receiver_cols[i].split('Receiver')[-1]}"
                        tipo_anomalia.append(tipo)

                moc = moc.with_columns([
                    pl.lit(transceiver).alias("Transceptor"),
                    pl.Series("Tipo", tipo_anomalia)
                ])
                registros.append(moc.select([
                    pl.col("time").alias("Timestamp"),
                    "Transceptor", "Tipo", "Tiempo"
                ]))

        if not registros:
            anomalies_global_moc = pl.DataFrame([])
            return [], "No se encontraron anomalías para las fechas seleccionadas."

        anomalies = pl.concat(registros).sort("Timestamp")
        anomalies_global_moc = anomalies

        transceptores = anomalies.select("Transceptor").unique().get_column("Transceptor").to_list()
        print(f"Procesamiento finalizado con {len(transceptores)} transceptores únicos.")
        return gr.update(choices=transceptores, visible=True), "Datos procesados exitosamente."

    except Exception as e:
        print("Error en procesar_datos_moc:", traceback.format_exc())
        anomalies_global_moc = None
        return [], str(e)

#Procesar datos de microliberación desde CSV
def procesar_datos_mlib(medidas):
    global anomalies_global_mlib
    try:
        print(f"Archivo recibido: {medidas.name}")

        extension = Path(medidas.name).suffix.lower()
        if extension == ".csv":
            datos = pl.read_csv(
                medidas.name,
                infer_schema_length=10000,
                schema_overrides={"value": pl.Float64},
                null_values=["", "NA", "null"],
                ignore_errors=True
            )
        elif extension == ".parquet":
            datos = pl.read_parquet(medidas.name)
        else:
            return None, "Formato de archivo no soportado."

        datos = datos.with_columns([
            pl.col("time").str.to_datetime(strict=False),
            pl.col("value").cast(pl.Float64, strict=False)
        ]).drop_nulls("time")

        registros = []

        transceivers = datos.select(pl.col("name").str.split("_").list.first()).unique().to_series().to_list()
        for transceiver in transceivers:
            datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{transceiver}_status"))
            if datos_filtrados.is_empty():
                continue

            pivot = datos_filtrados.pivot(index="time", values="value", columns="name", aggregate_function="first").sort("time")

            status_col = f"{transceiver}_status"
            receiver_cols = [col for col in pivot.columns if col.startswith(f"{transceiver}_statusReceiver")]

            df = pivot.with_columns((pl.col(status_col) - pl.col(status_col).shift(1)).alias("change"))
            for r in receiver_cols:
                df = df.with_columns((pl.col(r) - pl.col(r).shift(1)).alias(f"change_{r}"))
            df = df.filter(pl.col("change") != 0.0)
            df = df.with_columns((pl.col("time").cast(pl.Datetime("ms")) - pl.col("time").cast(pl.Datetime("ms")).shift(1))
                .cast(pl.Duration("ms")).cast(pl.Int64).fill_null(0).alias("Tiempo"))

            mlib = df.filter((pl.col("change") == -1.0) & pl.col("Tiempo").is_between(250, 2000))

            if not mlib.is_empty():
                tipo_anomalia = []
                for row in mlib.iter_rows(named=True):
                    cambios = [row.get(f"change_{r}", 0) for r in receiver_cols]
                    if all(c == 0 for c in cambios):
                        tipo_anomalia.append("microliberación cascada")
                    else:
                        tipo = "desconocido"
                        for i, c in enumerate(cambios):
                            if c == -1:
                                tipo = f"receptor {receiver_cols[i].split('Receiver')[-1]}"
                        tipo_anomalia.append(tipo)
                mlib = mlib.with_columns([
                    pl.lit(transceiver).alias("Transceptor"),
                    pl.Series("Tipo", tipo_anomalia)
                ])
                registros.append(mlib.select([
                    pl.col("time").alias("Timestamp"),
                    "Transceptor", "Tipo", "Tiempo"
                ]))

        if not registros:
            anomalies_global_mlib = pl.DataFrame([])
            return [], "No se encontraron anomalías para las fechas seleccionadas."

        anomalies = pl.concat(registros).sort("Timestamp")
        anomalies_global_mlib = anomalies

        transceptores = anomalies.select("Transceptor").unique().get_column("Transceptor").to_list()
        print(f"Procesamiento finalizado con {len(transceptores)} transceptores únicos.")
        return gr.update(choices=transceptores, visible=True), "Datos procesados exitosamente."

    except Exception as e:
        print("Error en procesar_datos_mlib:", traceback.format_exc())
        anomalies_global_mlib = None
        return [], str(e)

def mostrar_datos_detallados(df, fichero, evt: gr.SelectData):
    global series_global
    try:
        # Obtener el índice de la fila seleccionada
        fila_index = evt.index[0]

        # Extraer la fila seleccionada del DataFrame
        fila = df.iloc[fila_index]

        # Obtener el código de la fila seleccionada
        codigo = fila['Transceptor']
        receptor = fila['Receptor']

        # Medida a consultar
        medida = f"{codigo}_voltageReceiver{receptor}"

        # Lectura dinámica del archivo de medidas CSV o Parquet
        extension = Path(fichero.name).suffix.lower()

        if extension == ".csv":
            datos = pl.read_csv(fichero.name, infer_schema_length=10000, ignore_errors=True)
        elif extension == ".parquet":
            datos = pl.read_parquet(fichero.name)
        else:
            return None, "Formato de archivo no soportado."
        datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{codigo}"))
        datos_filtrados = datos_filtrados.to_pandas()

        # Convertir la columna 'time' a formato datetime y eliminar los posibles "na"
        datos_filtrados['time'] = pd.to_datetime(datos_filtrados['time'], format='mixed', errors='coerce')
        datos_filtrados = datos_filtrados[datos_filtrados['time'].notna()]

        # Pivotar el DataFrame para tener 'time' como índice y 'name' como columnas
        datos_pivot = datos_filtrados.pivot_table(index='time', columns='name', values='value', aggfunc='first')

        # Convertir todos los valores a numéricos, errores como NaN
        datos_pivot = datos_pivot.apply(pd.to_numeric, errors='coerce')

        # Diccionario para almacenar las series de tiempo de las medidas
        series = {}

        df_medida = datos_filtrados[datos_filtrados['name'] == medida]
        if not df_medida.empty:
            df_medida['value'] = pd.to_numeric(df_medida['value'], errors='coerce')
            df_medida = df_medida.set_index('time')
            series[medida] = df_medida[['value']]

        if not series:
            return None, "No se encontraron datos para las medidas seleccionadas."

        fig, axs = plt.subplots(len(series), 1, figsize=(12, 4 * len(series)), sharex=True)

        if len(series) == 1:
            axs = [axs]

        colors = plt.cm.tab10.colors
        for i, (medida, df_medida) in enumerate(series.items()):
            axs[i].plot(df_medida.index, df_medida['value'], label=medida, color=colors[i % len(colors)], linewidth=2)
            axs[i].set_title(fila['Transceptor'])
            axs[i].set_xlabel("Tiempo")
            axs[i].set_ylabel("Valor")
            axs[i].set_ylim(0, 3)
            axs[i].legend()

        plt.tight_layout()
        series_global = series

        return gr.update(value=fig, visible=True), "Consulta realizada con éxito."

    except Exception as e:
        print("Error en mostrar_datos_detallados:", traceback.format_exc())
        return None, f"Error al consultar los datos: {str(e)}"

def mostrar_datos_detallados_moc(df, fichero, evt: gr.SelectData):
    series_global_moc= None
    try:
        # Obtener el índice de la fila seleccionada
        fila_index = evt.index[0]

        # Extraer la fila seleccionada del DataFrame
        fila = df.iloc[fila_index]

        # Obtener el timestamp y el código de la fila seleccionada
        indice = pd.Timestamp(fila['Timestamp'])
        codigo = fila['Transceptor']

        # Asegurarse de que el índice esté en UTC
        if indice.tzinfo is None:
            indice = indice.tz_localize('UTC')
        else:
            indice = indice.tz_convert('UTC')

        # Definir el rango de tiempo
        inicio = (indice - pd.Timedelta(minutes=5)).isoformat()
        fin = (indice + pd.Timedelta(minutes=5)).isoformat()

        # Medidas a consultar
        medidas = [
            f"{codigo}_status",
            f"{codigo}_statusReceiver1",
            f"{codigo}_statusReceiver2",
            f"{codigo}_voltageReceiver1",
            f"{codigo}_voltageReceiver2",
        ]

        # Lectura dinámica del archivo de medidas CSV o Parquet
        extension = Path(fichero.name).suffix.lower()

        if extension == ".csv":
            datos = pl.read_csv(fichero.name, infer_schema_length=10000, ignore_errors=True)
        elif extension == ".parquet":
            datos = pl.read_parquet(fichero.name)
        else:
            return None, "Formato de archivo no soportado."
        datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{codigo}"))
        datos_filtrados = datos_filtrados.to_pandas()
        datos_filtrados['time'] = pd.to_datetime(datos_filtrados['time'], format='mixed', errors='coerce')
        datos_filtrados = datos_filtrados[datos_filtrados['time'].notna()]

        # Aplicar el filtro de tiempo
        inicio_dt = pd.to_datetime(inicio)
        fin_dt = pd.to_datetime(fin)
        datos_filtrados = datos_filtrados[(datos_filtrados['time'] >= inicio_dt) & (datos_filtrados['time'] <= fin_dt)]

        # Pivotar el DataFrame para tener 'time' como índice y 'name' como columnas
        datos_pivot = datos_filtrados.pivot_table(index='time', columns='name', values='value', aggfunc='first')

        # Convertir todos los valores a numéricos, errores como NaN
        datos_pivot = datos_pivot.apply(pd.to_numeric, errors='coerce')

        # Diccionario para almacenar las series de tiempo de las medidas
        series = {}

        for medida in medidas:
            df_medida = datos_filtrados[datos_filtrados['name'] == medida]
            if not df_medida.empty:
                df_medida['value'] = pd.to_numeric(df_medida['value'], errors='coerce')
                df_medida = df_medida.set_index('time')
                series[medida] = df_medida[['value']]

        if not series:
            return None, "No se encontraron datos para las medidas seleccionadas."


        # Separar medidas por tipo
        voltage_measures = {k: v for k, v in series.items() if "voltageReceiver" in k}
        status_measures = {k: v for k, v in series.items() if "status" in k}

        # Crear subgráficas
        num_status = len(status_measures)

        fig, axs = plt.subplots(1 + num_status, 1, figsize=(12, 4 + 2 * num_status), sharex=True,
                                 gridspec_kw={
                                     'height_ratios': [3] + [1] * num_status
                                 })

        if 1 + num_status == 1:
            axs = [axs]  # Asegurarse de que axs sea iterable si hay solo un gráfico

        # Graficar medidas de voltaje con estilo personalizado
        ax_voltage = axs[0]
        for medida, df_medida in voltage_measures.items():
            receptor_match = re.search(r'Receiver\d+', medida)
            receptor = receptor_match.group() if receptor_match else ""
            color = COLOR_MAP.get(receptor, {}).get("voltage", "gray")
            ax_voltage.plot(df_medida.index, df_medida['value'], label=medida, color=color, linewidth=2)
        ax_voltage.set_title("Medidas de Voltaje")
        ax_voltage.set_xlabel("Tiempo")
        ax_voltage.set_ylabel("Valor")
        ax_voltage.legend()

        # Graficar medidas de estado
        for ax, (medida, df_medida) in zip(axs[1:], status_measures.items()):
            receptor_match = re.search(r'Receiver\d+', medida)
            receptor = receptor_match.group() if receptor_match else "status"
            color = COLOR_MAP.get(receptor, COLOR_MAP.get("status", {})).get("status", "gray")
            ax.plot(df_medida.index, df_medida['value'], label=medida, drawstyle='steps-post', color=color)
            ax.fill_between(df_medida.index, 0, df_medida['value'], step='post', alpha=0.4, color=color)
            ax.set_title(medida)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Ocupado", "Libre"])
            ax.set_ylabel("")
            ax.legend()

        plt.tight_layout()
        series_global_moc = series
        return gr.update(value=fig, visible=True), "Consulta realizada con éxito."

    except Exception as e:
        print("Error en mostrar_datos_detallados:", traceback.format_exc())
        return None, f"Error al consultar los datos: {str(e)}"

def mostrar_datos_detallados_mlib(df, fichero, evt: gr.SelectData):
    series_global_mlib= None
    try:
        # Obtener el índice de la fila seleccionada
        fila_index = evt.index[0]

        # Extraer la fila seleccionada del DataFrame
        fila = df.iloc[fila_index]

        # Obtener el timestamp y el código de la fila seleccionada
        indice = pd.Timestamp(fila['Timestamp'])
        codigo = fila['Transceptor']

        # Asegurarse de que el índice esté en UTC
        if indice.tzinfo is None:
            indice = indice.tz_localize('UTC')
        else:
            indice = indice.tz_convert('UTC')

        # Definir el rango de tiempo
        inicio = (indice - pd.Timedelta(minutes=5)).isoformat()
        fin = (indice + pd.Timedelta(minutes=5)).isoformat()

        # Medidas a consultar
        medidas = [
            f"{codigo}_status",
            f"{codigo}_statusReceiver1",
            f"{codigo}_statusReceiver2",
            f"{codigo}_voltageReceiver1",
            f"{codigo}_voltageReceiver2",
        ]

        # Lectura dinámica del archivo de medidas CSV o Parquet
        extension = Path(fichero.name).suffix.lower()
        if extension == ".csv":
            datos = pl.read_csv(fichero.name, infer_schema_length=10000, ignore_errors=True)
        elif extension == ".parquet":
            datos = pl.read_parquet(fichero.name)
        else:
            return None, "Formato de archivo no soportado."
        datos_filtrados = datos.filter(pl.col("name").str.starts_with(f"{codigo}"))
        datos_filtrados = datos_filtrados.to_pandas()

        # Convertir la columna 'time' a formato datetime y eliminar los posibles "na"
        datos_filtrados['time'] = pd.to_datetime(datos_filtrados['time'], format='mixed', errors='coerce')
        datos_filtrados = datos_filtrados[datos_filtrados['time'].notna()]

        # Aplicar el filtro de tiempo
        inicio_dt = pd.to_datetime(inicio)
        fin_dt = pd.to_datetime(fin)
        datos_filtrados = datos_filtrados[(datos_filtrados['time'] >= inicio_dt) & (datos_filtrados['time'] <= fin_dt)]

        # Pivotar el DataFrame para tener 'time' como índice y 'name' como columnas
        datos_pivot = datos_filtrados.pivot_table(index='time', columns='name', values='value', aggfunc='first')

        # Convertir todos los valores a numéricos, errores como NaN
        datos_pivot = datos_pivot.apply(pd.to_numeric, errors='coerce')

        # Diccionario para almacenar las series de tiempo de las medidas
        series = {}

        for medida in medidas:
            df_medida = datos_filtrados[datos_filtrados['name'] == medida]
            if not df_medida.empty:
                df_medida['value'] = pd.to_numeric(df_medida['value'], errors='coerce')
                df_medida = df_medida.set_index('time')
                series[medida] = df_medida[['value']]

        if not series:
            return None, "No se encontraron datos para las medidas seleccionadas."


        # Separar medidas por tipo
        voltage_measures = {k: v for k, v in series.items() if "voltageReceiver" in k}
        status_measures = {k: v for k, v in series.items() if "status" in k}

        # Crear subgráficas
        num_status = len(status_measures)

        fig, axs = plt.subplots(1 + num_status, 1, figsize=(12, 4 + 2 * num_status), sharex=True,
                                 gridspec_kw={
                                     'height_ratios': [3] + [1] * num_status
                                 })

        if 1 + num_status == 1:
            axs = [axs]  # Asegurarse de que axs sea iterable si hay solo un gráfico

        # Graficar medidas de voltaje con estilo personalizado
        ax_voltage = axs[0]
        for medida, df_medida in voltage_measures.items():
            receptor_match = re.search(r'Receiver\d+', medida)
            receptor = receptor_match.group() if receptor_match else ""
            color = COLOR_MAP.get(receptor, {}).get("voltage", "gray")
            ax_voltage.plot(df_medida.index, df_medida['value'], label=medida, color=color, linewidth=2)
        ax_voltage.set_title("Medidas de Voltaje")
        ax_voltage.set_xlabel("Tiempo")
        ax_voltage.set_ylabel("Valor")
        ax_voltage.legend()

        # Graficar medidas de estado
        for ax, (medida, df_medida) in zip(axs[1:], status_measures.items()):
            receptor_match = re.search(r'Receiver\d+', medida)
            receptor = receptor_match.group() if receptor_match else "status"
            color = COLOR_MAP.get(receptor, COLOR_MAP.get("status", {})).get("status", "gray")
            ax.plot(df_medida.index, df_medida['value'], label=medida, drawstyle='steps-post', color=color)
            ax.fill_between(df_medida.index, 0, df_medida['value'], step='post', alpha=0.4, color=color)
            ax.set_title(medida)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Ocupado", "Libre"])
            ax.set_ylabel("")
            ax.legend()

        plt.tight_layout()
        series_global_mlib = series
        return gr.update(value=fig, visible=True), "Consulta realizada con éxito."

    except Exception as e:
        print("Error en mostrar_datos_detallados:", traceback.format_exc())
        return None, f"Error al consultar los datos: {str(e)}"

# Graficar Anomalías para la pestaña de bloqueos
def graficar_anomalías_bloqueos(transceivers_seleccionados):
    global anomalies_global_bloqueos
    if anomalies_global_bloqueos is None or anomalies_global_bloqueos.is_empty():
        return None, "No hay datos para mostrar."
    if not transceivers_seleccionados:
        return None, "Por favor selecciona al menos un transceptor."

    df = anomalies_global_bloqueos.filter(
        pl.col("Transceptor").is_in(transceivers_seleccionados)
    ).to_pandas()

    df = df.sort_values("Timestamp")

    plt.figure(figsize=(18, 9))
    for transceptor in transceivers_seleccionados:
        subset = df[df["Transceptor"] == transceptor]
        plt.scatter(subset["Timestamp"], subset["Tiempo"], label=transceptor, alpha=0.7)

    plt.title("Anomalías Transceptores Seleccionados")
    plt.xlabel("Tiempo")
    plt.ylabel("Duración (segundos)")
    plt.legend(title="Transceptores", loc="lower center", ncol=10, bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    plt.tight_layout()

    return gr.update(value=df, visible=True), gr.update(value=plt.gcf(), visible=True)

# Graficar Anomalías Microocupaciones
def graficar_anomalías_moc(transceivers_seleccionados):
    global anomalies_global_moc
    if anomalies_global_moc is None or anomalies_global_moc.is_empty():
        return None, "No hay datos para mostrar."

    if not transceivers_seleccionados:
        return None, "Por favor selecciona al menos un transceptor."

    # Filtrar con polars y convertir a pandas
    df = anomalies_global_moc.filter(
        pl.col("Transceptor").is_in(transceivers_seleccionados)
    ).to_pandas()

    df = df.sort_values("Timestamp")

    # Crear gráfico
    plt.figure(figsize=(18, 9))
    for transceptor in transceivers_seleccionados:
        subset = df[df["Transceptor"] == transceptor]
        plt.scatter(subset["Timestamp"], subset["Tiempo"], label=transceptor, alpha=0.7)

    plt.title("Anomalías Transceptores Seleccionados")
    plt.xlabel("Tiempo")
    plt.ylabel("Duración (milisegundos)")
    plt.legend(title="Transceptores", loc="lower center", ncol=10, bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    plt.tight_layout()

    return gr.update(value=df, visible=True), gr.update(value=plt.gcf(), visible=True)

# Graficar Anomalías Microliberaciones
def graficar_anomalías_mlib(transceivers_seleccionados):
    global anomalies_global_mlib
    if anomalies_global_mlib is None or anomalies_global_mlib.is_empty():
        return None, "No hay datos para mostrar."

    if not transceivers_seleccionados:
        return None, "Por favor selecciona al menos un transceptor."

    df = anomalies_global_mlib.filter(
        pl.col("Transceptor").is_in(transceivers_seleccionados)
    ).to_pandas()

    df = df.sort_values("Timestamp")

    plt.figure(figsize=(18, 9))
    for transceptor in transceivers_seleccionados:
        subset = df[df["Transceptor"] == transceptor]
        plt.scatter(subset["Timestamp"], subset["Tiempo"], label=transceptor, alpha=0.7)

    plt.title("Anomalías por Transceptores Seleccionados")
    plt.xlabel("Tiempo")
    plt.ylabel("Duración (milisegundos)")
    plt.legend(title="Transceptores", loc="lower center", ncol=10, bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    plt.tight_layout()

    return gr.update(value=df, visible=True), gr.update(value=plt.gcf(), visible=True)

# Interfaz de Gradio
# Estilo para deshabilitar la entrada del chatbot

css = """
#chatbox_read_only .input-container {
    display: none !important;
}
#graph_output_ia {
    width: 100% !important;
}
#table_output_ia {
    width: 100% !important;
}
"""
with gr.Blocks(css=css) as interfaz:
    gr.Markdown("# Explorador de anomalías en datos SAMTTC")
    with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("Sube un archivo `csv` o `parquet` con las medidas de los transceptores descargadas de un SAMTTC\n\n Una vez esté cargado el fichero que se desee analizar sólo es necesario seleccionar el tipo de anomalías que se quieran buscar en las pestañas que aparecen más abajo y presionar el botón procesar, o bien seleccionar la pestaña IA para realizar un análisis libre con soporte de IA")
            with gr.Column():
                medidas = gr.File(label="Subir archivo CSV o Parquet con datos de medidas", file_types=[".csv", ".parquet"])
    
    with gr.Tab("IA"):
        gr.Markdown("Interfaz de lenguaje natural a SQL para analizar el archivo introducido. \n\n Una vez traducida la consulta se puede ejecutar pulsando sobre `Ejecutar Consulta`")
        with gr.Row():    
            with gr.Column():    
                question_input_ia = gr.Textbox(label="Pregunta en lenguaje natural", placeholder="¿Voltaje del receptor 1 del circuito CV123?")
                ask_btn_ia = gr.Button("Traducir")
                query_output_ia = gr.Code(label="Consulta SQL", language="sql", lines=4, interactive=True)
                run_query_btn = gr.Button("Ejecutar Consulta")
        with gr.Row():
            with gr.Column():
                graph_output_ia = gr.Plot(label="Gráfico de resultado", elem_id="graph_output_ia", visible=False)
                table_output_ia = gr.Dataframe(label="Tabla de resultado", elem_id="table_output_ia", visible=False)
                text_output_ia = gr.Textbox(label="Resultado")

    with gr.Tab("Caídas"):
        gr.Markdown("Para buscar caídas de tensión o cambios pronunciados en el valor de libre que puedan provocar casos ocupación intempestiva por la caída de tensión de uno de los receptores. Extraemos los datos de cada uno de los receptores, calculamos la media y el valor máximo. La diferencia entre la media de los valores sin ocupación y el máximo nos debe dar como resultado algunos circuitos en los que veamos una caída de tensión muy pronunciada")    
        with gr.Row(equal_height=True):
            with gr.Column():
                boton_procesar = gr.Button("Procesar Datos")
            with gr.Column():
                mensaje_estado = gr.Textbox(label="Estado")
        with gr.Row():
            with gr.Column():
                tabla_salida = gr.DataFrame(label="Anomalías", visible=False)
                grafico_detallado = gr.Plot(visible=False)

    with gr.Tab("Bloqueos"):
        gr.Markdown("Para distinguir los casos de bloqueo del transceptor, buscamos el tiempo transcurrido entre medidas y buscamos que sea mayor o igual que tres minutos (180 seg).")    
        with gr.Row(equal_height=True):
            with gr.Column():
                boton_procesar_bloqueo = gr.Button("Procesar Datos")
            with gr.Column():
                mensaje_estado_bloqueo = gr.Textbox(label="Estado")
        with gr.Row():
            with gr.Column():
                transceivers_selector_bloqueo = gr.Dropdown(label="Selecciona Transceptores", choices=[], multiselect=True, allow_custom_value=True, visible=False)
                tabla_salida_bloqueo = gr.DataFrame(label="Anomalías", visible=False)
                grafico_detallado_bloqueo = gr.Plot(visible=False)

    with gr.Tab("Microocupaciones"):
        gr.Markdown("Para distinguir los casos de microocupación del resto lo que hacemos es iterar por todas las medidas con todos los valores de estado de los circuitos. Al dataframe se le añade el campo change que nos dirá cuándo hay un cambio de 1 a 0 (-1.0, ocupación) o de 0 a 1 (1.0, liberación) o si no hay cambio (0.0). Una vez marcados los cambios quitamos todos los valores que tengan en la columna change un 0.0 y calculamos en una tercera columna el valor del tiempo entre valores. Luego sólo nos queda filtrar los casos en los que haya alguna fila que cumpla la condición de que el cambio sea una liberación y el tiempo esté en el margen de 250ms-3000ms.")
        with gr.Row(equal_height=True):
            with gr.Column():
                boton_procesar_moc = gr.Button("Procesar Datos")
            with gr.Column():
                mensaje_estado_moc = gr.Textbox(label="Estado")
        with gr.Row():
            with gr.Column():
                transceivers_selector_moc = gr.Dropdown(label="Selecciona Transceptores", choices=[], multiselect=True, allow_custom_value=True, visible=False)
                grafico_salida_moc = gr.Plot(visible=False)
                tabla_salida_moc = gr.DataFrame(label="Anomalías", visible=False)
                grafico_detallado_moc = gr.Plot(visible=False)

    with gr.Tab("Microliberaciones"):
        gr.Markdown("Para distinguir los casos de microliberación del resto lo que hacemos es iterar por todas las medidas con todos los valores de estado de los circuitos. Al dataframe se le añade el campo change que nos dirá cuándo hay un cambio de 1 a 0 (-1.0, ocupación) o de 0 a 1 (1.0, liberación) o si no hay cambio (0.0). Una vez marcados los cambios quitamos todos los valores que tengan en la columna change un 0.0 y calculamos en una tercera columna el valor del tiempo entre valores. Luego sólo nos queda filtrar los casos en los que haya alguna fila que cumpla la condición de que el cambio sea una ocupación y el tiempo esté en el margen de 250ms-3000ms.")
        with gr.Row(equal_height=True):
            with gr.Column():
                boton_procesar_mlib = gr.Button("Procesar Datos")
            with gr.Column():
                mensaje_estado_mlib = gr.Textbox(label="Estado")
        with gr.Row():
            with gr.Column():
                transceivers_selector_mlib = gr.Dropdown(label="Selecciona Transceptores", choices=[], multiselect=True, allow_custom_value=True, visible=False)
                grafico_salida_mlib = gr.Plot(visible=False)
                tabla_salida_mlib = gr.DataFrame(label="Anomalías", visible=False)
                grafico_detallado_mlib = gr.Plot(visible=False)

    # Lógica de interacción IA

    ask_btn_ia.click(
        fn=ask_polars,
        inputs=[question_input_ia, medidas],
        outputs=[query_output_ia]
    )

    run_query_btn.click(
        fn=run_polars_query,
        inputs=[query_output_ia, medidas],
        outputs=[graph_output_ia, table_output_ia, text_output_ia]
    )

    # Lógica de interacción caídas

    boton_procesar.click(
        fn=procesar_datos,
        inputs=[medidas],
        outputs=[tabla_salida, mensaje_estado]
    )
    tabla_salida.select(
        fn=mostrar_datos_detallados,
        inputs=[tabla_salida, medidas],
        outputs=[grafico_detallado, mensaje_estado]
    )

    # Lógica de interacción bloqueos
    boton_procesar_bloqueo.click(
        fn=procesar_datos_bloqueo,
        inputs=[medidas],
        outputs=[transceivers_selector_bloqueo, mensaje_estado_bloqueo]
    )
    transceivers_selector_bloqueo.change(
        fn=graficar_anomalías_bloqueos,
        inputs=[transceivers_selector_bloqueo],
        outputs=[tabla_salida_bloqueo, grafico_detallado_bloqueo]
    )

    # Lógica de interacción microocupaciones
    boton_procesar_moc.click(
        fn=procesar_datos_moc,
        inputs=[medidas],
        outputs=[transceivers_selector_moc, mensaje_estado_moc]
    )
    transceivers_selector_moc.change(
        fn=graficar_anomalías_moc,
        inputs=[transceivers_selector_moc],
        outputs=[tabla_salida_moc, grafico_salida_moc]
    )
    tabla_salida_moc.select(
        fn=mostrar_datos_detallados_moc,
        inputs=[tabla_salida_moc, medidas],
        outputs=[grafico_detallado_moc, mensaje_estado_moc]
    )

    # Lógica de interacción microliberaciones
    boton_procesar_mlib.click(
        fn=procesar_datos_mlib,
        inputs=[medidas],
        outputs=[transceivers_selector_mlib, mensaje_estado_mlib]
    )
    transceivers_selector_mlib.change(
        fn=graficar_anomalías_mlib,
        inputs=[transceivers_selector_mlib],
        outputs=[tabla_salida_mlib, grafico_salida_mlib]
    )
    tabla_salida_mlib.select(
        fn=mostrar_datos_detallados_mlib,
        inputs=[tabla_salida_mlib, medidas],
        outputs=[grafico_detallado_mlib, mensaje_estado_mlib]
    )

    with gr.Tab("CSV2Parquet"):
        gr.Markdown("## Conversión de múltiples CSV a un único archivo Parquet\nSube uno o más archivos CSV, se unirán y transformarán para exportar como Parquet.\n\nTransformación: unir columnas `ttc_name` y `measurement` en una nueva columna `name`, eliminar filas donde `ttc_channel` == 'b' y conservar sólo las columnas `name`, `time`, `value`.")
        csv_files_input = gr.File(label="Subir archivos CSV", file_types=[".csv"], file_count="multiple")
        boton_convertir_parquet = gr.Button("Convertir y descargar Parquet")
        salida_parquet = gr.File(label="Archivo Parquet generado")

        def convertir_csvs_a_parquet(archivos):
            try:
                dfs = []
                for archivo in archivos:
                    df = pl.read_csv(archivo.name)
                    df = df.with_columns([
                        (pl.col("ttc_name").cast(pl.Utf8) + "_" + pl.col("measurement")).alias("name")
                    ])
                    df = df.filter(pl.col("ttc_channel") != "b")
                    df = df.select(["name", "time", "value"])
                    dfs.append(df)
                df_final = pl.concat(dfs)
                output_path = "/tmp/convertido.parquet"
                df_final.write_parquet(output_path)
                return output_path
            except Exception as e:
                print("Error en convertir_csvs_a_parquet:", e)
                return None

        boton_convertir_parquet.click(
            fn=convertir_csvs_a_parquet,
            inputs=[csv_files_input],
            outputs=[salida_parquet]
        )

# Iniciar la aplicación
if __name__ == "__main__":
    interfaz.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=("admin","admin")
    )