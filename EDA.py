# ------------------------------------ UNDER CONSTRUCTION------------------------------------
# Not not functional
# -------------------------------------------------------------------------------------------

#!/usr/bin/env python
# coding: utf-8

# # EDA Viajando: antes vs después de la pandemia en España  
# Proyecto de análisis de datos sobre la evolución de los viajes en el contexto post-pandémico en España.

# ![camera-1130731_1280.jpg](attachment:camera-1130731_1280.jpg)

# ## Introducción
# 
# La pandemia de la COVID-19 cambió radicalmente nuestras vidas y también nuestras ganas de movernos. Durante meses, las restricciones sanitarias mantuvieron a mucha gente en casa. Pero, ¿qué pasó cuando volvimos a salir? ¿Viajamos más? ¿Lo hicimos de forma diferente? ¿Hemos recuperado nuestro espíritu viajero?
# 
# Este proyecto de análisis exploratorio de datos (EDA) nace con la intención de entender cómo ha cambiado el comportamiento de los viajeros en España desde que comenzó la pandemia. Para ello, utilizaremos datos oficiales procedentes de fuentes como la Encuesta de Turismo de Residentes (ETR) y Estadística de Movimientos Turísticos en Fronteras (FRONTUR), con los que analizaremos las tendencias antes y después del año 2020.
# 
# Este análisis no solo trata de números, sino también de entender cómo ha evolucionado nuestra forma de viajar. A lo largo del proyecto, exploraremos estos temas con gráficos, comparaciones y datos visuales que nos ayudarán a responder cada hipótesis de forma clara y visual.
# 
# ### Prepárate para un viaje a través de los datos. ¡Empezamos!
# 
# 

# 
# ## Hipótesis
# 
# Nos planteamos las siguientes preguntas clave:
# 
# 1. Se viaja más despúes de la _pandemia_? Viajes de residentes antes y después de la pandemia.
#     - En España, el número de viajes ha aumentado tras la pandemia (2020).
# 2. Se viaja más por ocio que por trabajo? Motivos de viaje: _turismo_ vs _negocios_
#     - El turismo nacional de ocio ha aumentado más que los viajes por otros motivos (trabajo, estudios, etc.).  
#     - Utilizar la variable «Motivo del viaje» en la ETR para segmentar.
# 3. Influyen los festivos en el número de viajes? Las estaciones (tendencias _estacionales_)?
# 4. Los españoles viajan más dentro de España o prefieren el extranejo después de la pandemia?
# 5. El turismo internacional se ha recuperado más rápidamente que el nacional.
#     - Comprobar si los turistas extranjeros regresaron en mayor número o antes que los residentes nacionales.
#     - Comparar la evolución año a año desde 2020.
# 6. Tras la pandemia, los destinos costeros se han convertido en la opción preferida.
#     - Analizar por comunidad autónoma o provincia en ETR o FRONTUR si hay más viajes hacia zonas como Andalucía, Comunidad Valenciana, Canarias, Islas Baleares, etc.
# 7.  Ciertas comunidades autónomas han experimentado un mayor crecimiento turístico.
# 

# ## Datos necesarios  
# 
# * Volumen de viajes por año: `antes`, `durante` y `después`
# 
# * Segmentación de datos:
#     - país/ region
#     - mes/ año
#     - motivo del viaje: _ocio_, _trabajo_, _estudios_, _salud_, _visitas a familiares_.
# 
# * Eventos relevantes:  
#     - fechas clave: inicio de la pandemia, cuarentenas, levantamiento de restricciones

# ::: center
# | Variable                | Fuente recomendada         | Detalle                                                                 |
# |-------------------------|----------------------------|-------------------------------------------------------------------------|
# | Turistas internacionales| INE – Frontur              | Número de turistas por mes, país de origen                             |
# | Turistas nacionales     | INE – Movilidad / Egatur   | Desplazamientos por provincias / gasto medio                           |
# | Gasto turístico         | INE – Egatur               | Comparar gasto antes y después de la pandemia                          |
# | Transporte de viajeros  | INE – Transporte           | Avión, tren, bus; muestra la recuperación de la movilidad              |
# | Ocupación hotelera      | INE – Coyuntura turística  | Cuántas personas se alojan, noches, origen                             |
# | Fechas relevantes COVID | BOE / medios               | Fechas de restricciones/fin de estado de alarma                        |  
# 
# :::
# 

# ## Periodos a comparar
# 
# * __Antes de la pandemia__:     2018, 2019
# * __Durante pandemia__:         2020, 2021
# * __Post_pandemia__:            2022, 2023, 2024

# 
# ## Fuentes de datos
# 
# * [__INE__](https://www.ine.es/) : tiene datos mensuales de turistas
#     - __ETR__ - Encuesta de turismo de Residentes
#     - __FRONTUR__ - Turistas Internacionales por mes
#     - __Egatur__ -Gasto turistico
# * [__Datos Abiertos del Gobierno de España__](https://datos.gob.es)
#     - redirigen a INE
# * __Webscrapping__: [calendarios laborales por anios](https://www.calendarioslaborales.com)
# * __Eurostat__: Estadísticas de transporte y turismo para la UE.
# 

# ## Ideas
# 
# * Evolución anual del número de viajeros.
# * Comparativa 2019 vs. 2023/2024.
# * Distribución por CCAA.  
# * Impacto por CCAA, nacional vs. internacional.
# * Analizar el gasto turístico. Duración media del viaje y gasto.
# * Mapas de calor por región y año
# * Comparar con otros paises.
# 
# ## Referencias adicionales
# * [__Dataestur__](https://www.dataestur.es/): Plataforma que ofrece una selección de los datos más relevantes del turismo en España, con visualizaciones y cuadros de mando interactivos.
# * [text](https://datosmacro.expansion.com/comercio/turismo-internacional/espana)
# * [__EpData__](https://www.epdata.es/) : Portal que proporciona datos y gráficos sobre diversos aspectos del turismo en España, incluyendo transporte de viajeros y ocupación hotelera.
#     - [Transporte de viajeros](https://www.epdata.es/datos/transporte-viajeros-ine-datos-graficos/123/espana/106)
#     - [Ocupación hotelera](https://www.epdata.es/datos/ocupacion-hotelera-hoteles-datos-graficos/94/espana/106)

# ## Plan de trabajo
# 
# 1. __Descarga, limpieza y preparación de datos__ (eliminación de valores nulos, formato de fechas, etc.)
#     - Filtrar por anios clave: 2018, 2019
#     - Filtrar por motivos
#     - Agrupar por año -> número de viajes por turismo
# 2. __Análisis exploratorio__: 
# - Analizar gráficamente la evolución del turismo tras la pandemia mediante gráficos como:
#     + Evolución de viajes turísticos (línea por año)
#     + Comparativa: antes y después de la pandemia (barras 2019 vs. 2023)
#     + Reparto por tipo de viaje (solo si tienes más motivos en el dataset)
# - Explorar las tendencias en los datos, identificar patrones y comparar los diferentes periodos.
# 3. __Interpretación y validación de las hipótesis__:  
# Contestar: `¿se viaja más ahora?`  
#     - ¿Se han incrementado los viajes de ocio respecto a 2019?  
#     - ¿En qué año se recuperó?  
#     - ¿Hay diferencia entre los viajes nacionales y los internacionales?  
#     - ¿Hay algún mes o temporada que destaque?  
# 3. __Visualización de resultados y conclusiones__: mediante gráficos y tablas que ilustren claramente las diferencias en los patrones de viaje antes y después de la pandemia.

# ## Libraries & others musts

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class display(object):
    """Representador HTML de múltiples objetos"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


ccaa_con_costa = [
    'Andalucía',
    'Cataluña',
    'Comunitat Valenciana',
    'Murcia, Región de',
    'Galicia',
    'Cantabria',
    'País Vasco',
    'Asturias, Principado de',
    'Balears, Illes',
    'Canarias'
]


# ## Datasets para el análisis

# ### FRONTUR: extranjeros

# #### 13864 [Número de turistas según motivo principal del viaje](https://www.ine.es/jaxiT3/Tabla.htm?t=13864&L=0)
# - Descripción: Número de visitantes no residentes que acceden a España por las distintas vías de acceso.
# - Variables: Mes, motivo del viaje, total.
# 

# 
# ![alt](./img/13864.png)

# In[740]:


import pandas as pd

# *  Carga 13864.csv y limpieza

df_13864 = pd.read_csv("./data/13864.csv", sep=";")

# * Eliminamos la columna `Tipo de dato` dado que no aporta información
df_13864 = df_13864.drop('Tipo de dato', axis=1)

df_pivot_13864 = df_13864.pivot(index='Periodo', columns='Motivo del viaje', values='Total').reset_index()

# * Quitamos ``separadores`` de miles y convertir en ``numeros``
df_pivot_13864 = df_pivot_13864.replace('\.', '', regex=True)

col_no = [ 'Negocio, motivos profesionales', 'Ocio, recreo y vacaciones', 'Otros motivos', 'Total']
df_pivot_13864[col_no] = df_pivot_13864[col_no].apply(pd.to_numeric, errors='coerce')

df_pivot_13864['Mes'] = df_pivot_13864.Periodo.str[-2:]
df_pivot_13864['Anio'] = df_pivot_13864['Periodo'].str[:4]

nuevo_orden = ['Anio', 'Mes', 'Negocio, motivos profesionales',
       'Ocio, recreo y vacaciones', 'Otros motivos', 'Total']
df_pivot_13864 = df_pivot_13864[nuevo_orden]

# * Convertimos en ``int`` tmb el año y el mes
col_no = ['Anio', 'Mes']
df_pivot_13864[col_no] = df_pivot_13864[col_no].apply(pd.to_numeric, errors='coerce')
df_pivot_13864.info()

# * Ordenamos descendete anio, mes
df_pivot_13864 = df_pivot_13864.sort_values(by=['Anio', 'Mes'], ascending=[False, False])

df_pivot_13864.to_csv('./data/frt_13864_clean.csv')

display('df_pivot_13864.head()', 'df_pivot_13864.tail()')


# In[741]:


df_pivot_13864.info()


# ##### VIZ Echarts Viajes totales mensuales apilados por año

# In[742]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend
from ipecharts.option.series import Line
import pandas as pd

df = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)

meses = [str(m) for m in range(1, 13)]
anios = sorted(df['Anio'].unique())


# Para cada año creamos la serie con valores ordenados por mes
series = []
colores = [
    '#80FFA5',  # verde menta suave
    '#00DDFF',  # azul celeste
    '#37A2FF',  # azul brillante
    '#FF0087',  # rosa fuerte
    '#FFBF00',  # amarillo mostaza
    '#7FFF00',  # verde chartreuse
    '#1E90FF',  # azul dodger
    '#FF4500',  # rojo anaranjado
    '#FFD700',  # dorado
    '#00FA9A',  # verde medio mar
    '#BA55D3'   # púrpura medio
]

for i, año in enumerate(anios):
    df_año = df[df['Anio'] == año].set_index('Mes')
    y_data = [int(df_año.loc[m, 'Total']) if m in df_año.index else 0 for m in range(1, 13)]

    line = Line(
        name=str(año),
        type='line',
        data=y_data,
        stack='Total',
        smooth=True,
        showSymbol=False,
        lineStyle={'width': 0},
        areaStyle={'opacity': 0.8, 'color': colores[i]}
    )
    series.append(line)

# Ejes
xAxis = XAxis(type="category", data=meses)
yAxis = YAxis(type="value")

# Opción
option = Option()
option.title = Title(text="Viajes Totales mensuales apilados por Año")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=[str(a) for a in anios], top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colores
option.series = series

# Crear widget
chart = EChartsWidget()
chart.option = option
chart


# ##### VIZ Pyplot Viajes totales mensuales linear por año

# In[743]:


import pandas as pd
import matplotlib.pyplot as plt

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)

# Colores para las líneas
colors = [
    '#80FFA5', '#00DDFF', "#08090A", '#FF0087', '#FFBF00',
    '#7FFF00', '#1E90FF', '#FF4500', '#FFD700', '#00FA9A', '#BA55D3'
]

# Preparamos el gráfico
plt.figure(figsize=(14,8))

# mark area confinamiento
for idx, anio in enumerate(anios):
    if anio == 2020:
        start = idx - 1  # abril
        end = idx + 1    # junio
        plt.axvspan(start, end, color='red', alpha=0.1)
        break

# Agrupamos por año para trazar líneas por año
for i, (year, group) in enumerate(df_pivot_13864.groupby('Anio')):
    x = group['Mes']
    y = group['Total']
    plt.plot(x, y, label=str(year), color=colors[i], marker='o')

plt.title('Total viajes por mes y año')
plt.xlabel('Mes')
plt.ylabel('Total viajes')
plt.legend(title='Año')
plt.grid(False)
plt.xticks(range(1,13))
plt.savefig('./img/viz-fro-total-per-mes-lineal.png')
plt.show()


# ##### VIZ Echarts Viajes por mes (2015–2025) con confinamiento destacado 

# In[744]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Line

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)

months = list(map(str, range(1, 13)))

colors = [
    '#80FFA5', '#00DDFF', '#37A2FF', '#FF0087', '#FFBF00',
    '#7FFF00', '#1E90FF', '#FF4500', '#FFD700', '#00FA9A', '#BA55D3'
]

series = []
legend_names = []

# Definimos el área marcada (marzo a junio)
mark_area = {
    'data': [[{'xAxis': '4'}, {'xAxis': '6'}]],
    'itemStyle': {
        'color': 'rgba(255, 173, 177, 0.2)'  # rosa claro semitransparente
    }
}

for i, year in enumerate(range(2015, 2026)):
    df_year = df_pivot_13864[df_pivot_13864['Anio'] == year]
    df_year = df_year.groupby('Mes')['Total'].sum().reindex(months, fill_value=np.nan).reset_index()

    line = Line(
        name=str(year),
        type="line",
        data=df_year['Total'].tolist(),
        smooth=True,
        showSymbol=True,
        lineStyle={'width': 2},
        areaStyle=None
    )

    # Añadimos el markArea solo al año 2020 o a la primera serie si lo quieres siempre visible
    if year == 2020:
        line.markArea = mark_area

    series.append(line)
    legend_names.append(str(year))

chart = EChartsWidget()

xAxis = XAxis(type="category", data=months)
yAxis = YAxis(type="value")

option = Option()
option.title = Title(text="Viajes por mes (2015–2025)")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=legend_names, top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colors[:len(series)]
option.series = series
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})

chart.option = option
chart


# ##### VIZ Echarts Comparación viajes por intervalos de 2 años
# 
# * **Antes**: 2018 -2019
# * **Durante**: 2020 -2021
# * **Despues 1**: 2022 -2023
# * **Despues 2**: 2024 -2025

# In[746]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Line

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)

# Datos: meses 1-12 (string) para eje X
months = list(map(str, range(1, 13)))

# Agrupar por mes, sumando el total mensual
totales_mens_18_19 = df_pivot_13864[df_pivot_13864['Anio'].isin([2018, 2019])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_20_21 = df_pivot_13864[df_pivot_13864['Anio'].isin([2020, 2021])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_22_23 = df_pivot_13864[df_pivot_13864['Anio'].isin([2022, 2023])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_24_25 = df_pivot_13864[df_pivot_13864['Anio'].isin([2024, 2025])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()

# Datos de cada año (por mes)
data_18_19 = totales_mens_18_19.Total.values
data_20_21 = totales_mens_20_21.Total.values
data_22_23 = totales_mens_22_23.Total.values
data_24_25 = totales_mens_24_25.Total.values

chart = EChartsWidget()

xAxis = XAxis(type="category", data=months)
yAxis = YAxis(type="value")

colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

# Definimos el área marcada (marzo a junio)
mark_area = {
    'data': [[{'xAxis': '4'}, {'xAxis': '6'}]],
    'itemStyle': {
        'color': 'rgba(255, 173, 177, 0.2)'  # rosa claro semitransparente
    }
}

line1 = Line(name="2018-2019", type="line", data=data_18_19, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)
line2 = Line(name="2020-2021", type="line", data=data_20_21, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None, markArea=mark_area)
line3 = Line(name="2022-2023", type="line", data=data_22_23, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)
line4 = Line(name="2024-2025", type="line", data=data_24_25, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)

option = Option()

option.title = Title(text="FRO: Viajes por mes y agrupaciones de años")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=["2018-2019", "2020-2021", "2022-2023", "2024-2025"], top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colors
option.series = [line1, line2, line3, line4]
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})

chart.option = option
chart


# ##### VIZ Echarts Comparación de viajes mensual por intervalos 2 vs 3 años
# 
# * **Antes**: 2018 - 2019
# * **Durante**: 2020 - 2021
# * **Despues**: 2022 - 2023 - 2024
# 

# In[747]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Line

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)

# Datos: meses 1-12 (string) para eje X
months = list(map(str, range(1, 13)))

# Agrupar por mes, sumando el total mensual
totales_mens_18_19 = df_pivot_13864[df_pivot_13864['Anio'].isin([2018, 2019])].groupby([ 'Mes'])['Total'].sum().reset_index()
totales_mens_20_21 = df_pivot_13864[df_pivot_13864['Anio'].isin([2020, 2021])].groupby([ 'Mes'])['Total'].sum().reset_index()
totales_mens_22_23_24 = df_pivot_13864[df_pivot_13864['Anio'].isin([2022, 2023, 2024])].groupby([ 'Mes'])['Total'].sum().reset_index()

# Datos de cada año (por mes)
data_18_19 = totales_mens_18_19.Total.values
data_20_21 = totales_mens_20_21.Total.values
data_22_23_24 = totales_mens_22_23_24.Total.values

mark_area = {
    'data': [
        [{'xAxis': '4'}, {'xAxis': '6'}]  # Marca desde abril (5) a mayo (6)
    ],
    'itemStyle': {
        'color': 'rgba(255, 173, 177, 0.2)'
    }
}
chart = EChartsWidget()

xAxis = XAxis(type="category", data=months)
yAxis = YAxis(type="value")

colors = ['#80FFA5', '#FF0087', '#00DDFF']


line1 = Line(name="2018-2019", type="line", data=data_18_19, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None, markArea=mark_area)
line2 = Line(name="2020-2021", type="line", data=data_20_21, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)
line3 = Line(name="2022-2023-2024", type="line", data=data_22_23_24, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)

option = Option()
option.title = Title(text="FRO: Comparación de viajes mensual por intervalos 2 vs 3 años")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=["2018-2019", "2020-2021", "2022-2023-2024"], top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colors
option.series = [line1, line2, line3]
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})


chart.option = option
chart


# ##### VIZ Echarts Donut Distribución total de viajes por periodo

# In[749]:


import pandas as pd
from ipecharts import EChartsWidget
from ipecharts.option import Option, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Pie

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)

# df_pivot_13864.head()
# df_pivot_13864.info()
frpo1_suma_2018_2019 = df_pivot_13864[df_pivot_13864['Anio'].isin([2018, 2019])]['Total'].sum()
frpo1_suma_2020_2021 = df_pivot_13864[df_pivot_13864['Anio'].isin([2020, 2021])]['Total'].sum()
frpo1_suma_2022_2023 = df_pivot_13864[df_pivot_13864['Anio'].isin([2022, 2023])]['Total'].sum()
frpo1_suma_2024_2025 = df_pivot_13864[df_pivot_13864['Anio'].isin([2024, 2025])]['Total'].sum()

data = [
    {"value": frpo1_suma_2018_2019, "name": "2018–2019"},
    {"value": frpo1_suma_2020_2021, "name": "2020–2021"},
    {"value": frpo1_suma_2022_2023, "name": "2022–2023"},
    {"value": frpo1_suma_2024_2025, "name": "2024–2025"}
]

# Colores opcionales
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

# Crear widget de gráfico
chart = EChartsWidget()

# Crear serie tipo "donut"
series = Pie(
    name="Viajes por periodo",
    type="pie",
    radius=["40%", "70%"],
    avoidLabelOverlap=False,
    itemStyle={
        "borderRadius": 10,
        "borderColor": "#fff",
        "borderWidth": 2
    },
    label={
        "show": True,
        "position": "inside",
        "formatter": "{d}%",
        "fontSize": 12
    },
    emphasis={
        "label": {
            "show": True,
            "fontSize": 20,
            "fontWeight": "bold"
        }
    },
    labelLine={"show": False},
    data=data
)

# Configurar opciones del gráfico
option = Option()
option.color = colors
option.title = Title(text="FRO: Distribución total de viajes por periodo", left="center")
option.tooltip = Tooltip(trigger="item", formatter="{b}: {c} ({d}%)")
option.legend = Legend(top="5%", left="center")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})
option.series = [series]

# Asignar opción al gráfico
chart.option = option
chart


# ##### VIZ Echarts Donut por Tipo de viaje por intervalos de dos años

# In[751]:


import pandas as pd
from ipecharts import EChartsWidget
from ipecharts.option import Option, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Pie

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)
# df_pivot_13864.head()
# df_pivot_13864.info()
fro1_negocio_2018_2019 = df_pivot_13864[df_pivot_13864['Anio'].isin([2018, 2019])]['Negocio, motivos profesionales'].sum()
fro1_negocio_2020_2021 = df_pivot_13864[df_pivot_13864['Anio'].isin([2020, 2021])]['Negocio, motivos profesionales'].sum()
fro1_negocio_2022_2023 = df_pivot_13864[df_pivot_13864['Anio'].isin([2022, 2023])]['Negocio, motivos profesionales'].sum()
fro1_negocio_2024_2025 = df_pivot_13864[df_pivot_13864['Anio'].isin([2024, 2025])]['Negocio, motivos profesionales'].sum()

fro1_ocio_2018_2019 = df_pivot_13864[df_pivot_13864['Anio'].isin([2018, 2019])]['Ocio, recreo y vacaciones'].sum()
fro1_ocio_2020_2021 = df_pivot_13864[df_pivot_13864['Anio'].isin([2020, 2021])]['Ocio, recreo y vacaciones'].sum()
fro1_ocio_2022_2023 = df_pivot_13864[df_pivot_13864['Anio'].isin([2022, 2023])]['Ocio, recreo y vacaciones'].sum()
fro1_ocio_2024_2025 = df_pivot_13864[df_pivot_13864['Anio'].isin([2024, 2025])]['Ocio, recreo y vacaciones'].sum()

fro1_otros_2018_2019 = df_pivot_13864[df_pivot_13864['Anio'].isin([2018, 2019])]['Otros motivos'].sum()
fro1_otros_2020_2021 = df_pivot_13864[df_pivot_13864['Anio'].isin([2020, 2021])]['Otros motivos'].sum()
fro1_otros_2022_2023 = df_pivot_13864[df_pivot_13864['Anio'].isin([2022, 2023])]['Otros motivos'].sum()
fro1_otros_2024_2025 = df_pivot_13864[df_pivot_13864['Anio'].isin([2024, 2025])]['Otros motivos'].sum()

# Datos ya procesados
datos_motivos = {
    "Negocio": [
        {"value": fro1_negocio_2018_2019, "name": "2018–2019"},
        {"value": fro1_negocio_2020_2021, "name": "2020–2021"},
        {"value": fro1_negocio_2022_2023, "name": "2022–2023"},
        {"value": fro1_negocio_2024_2025, "name": "2024–2025"}
    ],
    "Ocio": [
        {"value": fro1_ocio_2018_2019, "name": "2018–2019"},
        {"value": fro1_ocio_2020_2021, "name": "2020–2021"},
        {"value": fro1_ocio_2022_2023, "name": "2022–2023"},
        {"value": fro1_ocio_2024_2025, "name": "2024–2025"}
    ],
    "Otros": [
        {"value": fro1_otros_2018_2019, "name": "2018–2019"},
        {"value": fro1_otros_2020_2021, "name": "2020–2021"},
        {"value": fro1_otros_2022_2023, "name": "2022–2023"},
        {"value": fro1_otros_2024_2025, "name": "2024–2025"}
    ]
}

# Crear widget de gráfico
chart = EChartsWidget()

# Colores para los períodos
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

# Crear lista de series para cada gráfico
series = []
for idx, (motivo, data) in enumerate(datos_motivos.items()):
    series.append(
        Pie(
            name=motivo,
            type="pie",
            radius=["30%", "60%"],
            center=[f"{(idx + 1) * 25}%", "50%"],
            avoidLabelOverlap=False,
            itemStyle={
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2
            },
            label={
                "show": True,
                "position": "center",
                "formatter": motivo,
                "fontSize": 16,
                "fontWeight": "bold"
            },
            emphasis={
                "label": {
                    "show": True,
                    "fontSize": 16,
                    "fontWeight": "bold"
                }
            },
            labelLine={"show": False},
            data=data
        )
    )

centers = ["20%", "50%", "80%"]

# Configurar opciones del gráfico
option = Option()
option.title = Title(text="FRO: Distribución de viajes por motivo y período", left="center")
option.tooltip = Tooltip(trigger="item", formatter="{a}<br/>{b}: {c} ({d}%)")
option.legend = Legend(orient="vertical", left="left")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})
option.series = series
option.color = colors

# Asignar opción al gráfico
chart.option = option
chart


# In[753]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Pie

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)

# Datos por motivo
datos_motivos = {
    "Negocio": [
        {"value": negocio_2018_2019, "name": "2018–2019"},
        {"value": negocio_2020_2021, "name": "2020–2021"},
        {"value": negocio_2022_2023, "name": "2022–2023"},
        {"value": negocio_2024_2025, "name": "2024–2025"}
    ],
    "Ocio": [
        {"value": ocio_2018_2019, "name": "2018–2019"},
        {"value": ocio_2020_2021, "name": "2020–2021"},
        {"value": ocio_2022_2023, "name": "2022–2023"},
        {"value": ocio_2024_2025, "name": "2024–2025"}
    ],
    "Otros": [
        {"value": otros_2018_2019, "name": "2018–2019"},
        {"value": otros_2020_2021, "name": "2020–2021"},
        {"value": otros_2022_2023, "name": "2022–2023"},
        {"value": otros_2024_2025, "name": "2024–2025"}
    ]
}

# Crear widget de gráfico
chart = EChartsWidget()

# Crear las series de tipo "donut"
series = []
centers = ["20%", "50%", "80%"]
motivos = list(datos_motivos.keys())
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

for i, motivo in enumerate(motivos):
    series.append(
        Pie(
            name=motivo,
            type="pie",
            radius=["30%", "60%"],
            center=[centers[i], "60%"],
            avoidLabelOverlap=False,
            itemStyle={
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2
            },
            label={
                "show": True,
                "position": "inside",
                "formatter": "{d}%",
                "fontSize": 12
            },
            emphasis={
                "label": {
                    "show": True,
                    "fontSize": 14,
                    "fontWeight": "bold"
                }
            },
            labelLine={"show": True},
            data=datos_motivos[motivo]
        )
    )

# Configurar opciones del gráfico
option = Option()
option.title = Title(text="FRO: Distribución de viajes por motivo y período", left="center", top="5%")
option.tooltip = Tooltip(trigger="item", formatter="{a}<br/>{b}: {c} ({d}%)")
option.legend = Legend(orient="vertical", left="left")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})
option.series = series
option.color = colors

# Asignar opción al gráfico
chart.option = option
chart


# In[754]:


import pandas as pd
from ipecharts import EChartsWidget
from ipecharts.option import Option, Tooltip, Legend, Grid, XAxis, YAxis, Title
from ipecharts.option.series import Bar

df_pivot_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_pivot_13864['Mes'] = df_pivot_13864['Mes'].astype(str)

# Agrupa por año y suma cada motivo
df_grouped = df_pivot_13864.groupby('Anio').agg({
    'Negocio, motivos profesionales': 'sum',
    'Ocio, recreo y vacaciones': 'sum',
    'Otros motivos': 'sum'
}).reset_index()

anios = df_grouped['Anio'].astype(str).tolist()
negocio = df_grouped['Negocio, motivos profesionales'].tolist()
ocio = df_grouped['Ocio, recreo y vacaciones'].tolist()
otros = df_grouped['Otros motivos'].tolist()

series = [
    Bar(name='Negocio', data=negocio, stack='total', label={'show': False}),
    Bar(name='Ocio', data=ocio, stack='total', label={'show': False}),
    Bar(name='Otros', data=otros, stack='total', label={'show': False}),
]

option = Option()
option.tooltip = Tooltip(trigger='axis', axisPointer={'type': 'shadow'})
option.legend = Legend(orient="horizontal", top="bottom")
option.grid = Grid(left='3%', right='4%', bottom='3%', containLabel=True)
option.xAxis = XAxis(type='value')
option.yAxis = YAxis(type='category', data=anios)
option.series = series
option.title = Title(text="FRO: Totales apilados por motivo de viaje y año", left="center")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})

chart = EChartsWidget()
chart.option = option
chart


# #### 13884 [Resultados detallados mensuales FRONTUR](https://www.ine.es/jaxiT3/Tabla.htm?t=13884)

# - Descripción: Número de visitantes no residentes que acceden a España por las distintas vías de acceso.
# - Variables: Mes, CCAA destino, motivo del viaje, total.

# <!-- ![Img](./img/13884.png) -->
# ![image.png](attachment:image.png)

# Al ser un dataset grande (unos 30 GB) me planteo descargarlo por años y gestionarlo por lotes.

# ** Verificar por que hay una ligera diferencia entre el total calculado del DS 13884 y el DS 13864:
# * DS 13884
# Motivo del viaje	Negocios y otros motivos profesionales	Ocio, recreo y vacaciones	Otros motivos	Total de motivos del viaje
# Periodo				
# 2024M11	388758.0	4534348.0	483009.0	5570619.0
# 
# 
# * DS 13864
# Periodo	Negocio, motivos profesionales	Ocio, recreo y vacaciones	Otros motivos	Total
# 109	2024M11	463667	4622268	584284	5670219
# 

# In[ ]:


import os
import pandas as pd

archivos = [f for f in os.listdir('./data') if '13884' in f]
dfs = []

for archivo in archivos:
    path_archivo = os.path.join('./data/', archivo)
    df = pd.read_csv(path_archivo, sep=";", encoding='latin-1')
    # df['año'] = año  # Agregar columna con el año (opcional)

    # Eliminar las cols con valores únicos
    unique_counts = df.describe(include='all').loc['unique']
    cols_v_unique = unique_counts[unique_counts == 1].index.tolist()
    df_filtrado = df.drop(columns=cols_v_unique)

    # Replace los separadores de miles '.' del Total
    df_filtrado.Total = df_filtrado.Total.replace('\.', '', regex=True)

    # Elimino los registros 'Total nacional'
    df_filtrado = df_filtrado[df_filtrado['Comunidad autónoma de destino'] != 'Total nacional']

    # Convert to number
    df_filtrado.Total = df_filtrado.Total.apply((pd.to_numeric), errors='coerce')

    # pivot para sacar el total de viajes por CCAA mensual
    df_filtrado_pivot_motivo = df_filtrado.pivot_table(
        index='Periodo',
        columns='Comunidad autónoma de destino',
        values='Total',
        aggfunc='sum'
    ).sort_index()

    dfs.append(df_filtrado_pivot_motivo)

# Unir todos los DataFrames en uno solo
df_13884_pivot_ccaa_total = pd.concat(dfs, ignore_index=False)

df_13884_pivot_ccaa_total.to_csv('./data/13884_15_25_ccaa.csv')

df_13884_pivot_ccaa_total.head()


# In[716]:


df_13884_pivot_ccaa_total.info()


# ### ETR (Residentes, nacionales)

# #### 12421 [Viajes por motivo principal del viaje](https://www.ine.es/jaxiT3/Tabla.htm?t=12421) 

# ![image.png](attachment:image.png)

# In[755]:


import pandas as pd

df_12421 = pd.read_csv('./data/12421.csv', sep=';', encoding='latin-1')
# display('df_12421.head()', 'df_12421.tail()')
# df_12421['Tipo de dato'].nunique()
# df_12421['Concepto turístico'].nunique()

# Borramos las columnas con valores únicos 
df_12421 = df_12421.drop(['Tipo de dato', 'Concepto turístico'], axis=1)

# df_12421['Motivo principal'].unique()
# Pivotamos la tabla
df_pivot_12421 = df_12421.pivot(index='Periodo', columns='Motivo principal', values='Total').reset_index()

# Replace separadores de miles '.'
df_pivot_12421 = df_pivot_12421.replace('\.', '', regex=True)

# Convertimos las cols en numerico
col_no = [ 'Negocios y otros motivos profesionales', 'Ocio, recreo y vacaciones', 'Otros motivos', 'Total', 'Visitas a familiares o amigos']
df_pivot_12421[col_no] = df_pivot_12421[col_no].apply(pd.to_numeric, errors='coerce')

# Separo la columna Periodo en Anio y Mes
df_pivot_12421['Mes'] = df_pivot_12421.Periodo.str[-2:]
df_pivot_12421['Anio'] = df_pivot_12421['Periodo'].str[:4]

# * Dado que en `FRONTUR 13864` los __Motivos principales__ que tenemos solo `Negocio, motivos profesionales`, 
# `Ocio, recreo y vacaciones` y `Otros motivos` vamos a englobar el grupo de `Visitas a familiares o amigos` 
# en el grupo de `Otros motivos`
df_pivot_12421['Otros motivos'] = df_pivot_12421['Otros motivos'] + df_pivot_12421['Visitas a familiares o amigos']
df_pivot_12421 = df_pivot_12421.drop('Visitas a familiares o amigos', axis=1)

nuevo_orden = ['Anio', 'Mes', 'Negocios y otros motivos profesionales',
       'Ocio, recreo y vacaciones', 'Otros motivos', 'Total']
df_pivot_12421 = df_pivot_12421[nuevo_orden]
df_pivot_12421

# Convertimos las cols en numerico
col_no = ['Anio', 'Mes']
df_pivot_12421[col_no] = df_pivot_12421[col_no].apply(pd.to_numeric, errors='coerce')
df_pivot_12421 = df_pivot_12421.sort_values(by=['Anio', 'Mes'], ascending=[False, False])

df_pivot_12421.to_csv('./data/ert_12421_clean.csv')
df_pivot_12421.info()


# ##### VIZ Pyplot Viajes totales mensuales linear por año

# In[761]:


import pandas as pd
import matplotlib.pyplot as plt

df_pivot_12421 = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)
anios = sorted(df_pivot_12421['Anio'].unique())

# Colores para las líneas
colors = [
    '#80FFA5', '#00DDFF', "#08090A", '#FF0087', '#FFBF00',
    '#7FFF00', '#1E90FF', '#FF4500', '#FFD700', '#00FA9A', '#BA55D3'
]

# Preparamos el gráfico
plt.figure(figsize=(14,8))

# mark area confinamiento
for idx, anio in enumerate(anios):
    if anio == 2020:
        start = idx - 1  # abril
        end = idx + 1    # junio
        plt.axvspan(start, end, color='red', alpha=0.1)
        break

# Agrupamos por año para trazar líneas por año
for i, (year, group) in enumerate(df_pivot_12421.groupby('Anio')):
    x = group['Mes']
    y = group['Total']
    plt.plot(x, y, label=str(year), color=colors[i], marker='o')

plt.title('ETR: Total viajes por mes y año')
plt.xlabel('Mes')
plt.ylabel('Total viajes')
plt.legend(title='Año')
plt.grid(False)
plt.xticks(range(1,13))
plt.savefig('./img/viz-etr-total-per-mes-lineal.png')
plt.show()


# ##### VIZ Echarts Viajes por mes (2015–2025) con confinamiento destacado 

# In[ ]:


import numpy as np
import warnings
warnings.filterwarnings("ignore")

from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Line

df_pivot_12421 = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)
df_pivot_12421['Mes'] = df_pivot_12421['Mes'].astype(str)

months = list(map(str, range(1, 13)))

colors = [
    '#80FFA5', '#00DDFF', '#37A2FF', '#FF0087', '#FFBF00',
    '#7FFF00', '#1E90FF', '#FF4500', '#FFD700', '#00FA9A', '#BA55D3'
]

series = []
legend_names = []

# Definimos el área marcada (marzo a junio)
mark_area = {
    'data': [[{'xAxis': '4'}, {'xAxis': '6'}]],
    'itemStyle': {
        'color': 'rgba(255, 173, 177, 0.2)'  # rosa claro semitransparente
    }
}

for i, year in enumerate(range(2015, 2026)):
    df_year = df_pivot_12421[df_pivot_12421['Anio'] == year]
    df_year = df_year.groupby('Mes')['Total'].sum().reindex(months, fill_value=np.nan).reset_index()

    line = Line(
        name=str(year),
        type="line",
        data=df_year['Total'].tolist(),
        smooth=True,
        showSymbol=True,
        lineStyle={'width': 2},
        areaStyle=None
    )

    # Añadimos el markArea solo al año 2020 
    if year == 2020:
        line.markArea = mark_area

    series.append(line)
    legend_names.append(str(year))

chart = EChartsWidget()

xAxis = XAxis(type="category", data=months)
yAxis = YAxis(type="value")

option = Option()
option.title = Title(text="ETR: Viajes por mes (2015–2025)")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=legend_names, top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colors[:len(series)]
option.series = series
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})

chart.option = option
chart


# ##### VIZ Echarts Comparación viajes por intervalos de 2 años
# 
# * **Antes**: 2018 -2019
# * **Durante**: 2020 -2021
# * **Despues 1**: 2022 -2023
# * **Despues 2**: 2024 -2025

# In[762]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Line

df_pivot_12421 = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)
df_pivot_12421['Mes'] = df_pivot_12421['Mes'].astype(str)

# Datos: meses 1-12 (string) para eje X
months = list(map(str, range(1, 13)))

# Agrupar por mes, sumando el total mensual
totales_mens_18_19 = df_pivot_12421[df_pivot_12421['Anio'].isin([2018, 2019])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_20_21 = df_pivot_12421[df_pivot_12421['Anio'].isin([2020, 2021])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_22_23 = df_pivot_12421[df_pivot_12421['Anio'].isin([2022, 2023])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_24_25 = df_pivot_12421[df_pivot_12421['Anio'].isin([2024, 2025])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()

# Datos de cada año (por mes)
data_18_19 = totales_mens_18_19.Total.values
data_20_21 = totales_mens_20_21.Total.values
data_22_23 = totales_mens_22_23.Total.values
data_24_25 = totales_mens_24_25.Total.values

chart = EChartsWidget()

xAxis = XAxis(type="category", data=months)
yAxis = YAxis(type="value")

colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

# Definimos el área marcada (abril a junio)
mark_area = {
    'data': [[{'xAxis': '4'}, {'xAxis': '6'}]],
    'itemStyle': {
        'color': 'rgba(255, 173, 177, 0.2)'  # rosa claro semitransparente
    }
}

line1 = Line(name="2018-2019", type="line", data=data_18_19, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)
line2 = Line(name="2020-2021", type="line", data=data_20_21, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None, markArea=mark_area)
line3 = Line(name="2022-2023", type="line", data=data_22_23, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)
line4 = Line(name="2024-2025", type="line", data=data_24_25, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)

option = Option()

option.title = Title(text="ETR: Viajes por mes y agrupaciones de años")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=["2018-2019", "2020-2021", "2022-2023", "2024-2025"], top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colors
option.series = [line1, line2, line3, line4]
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})

chart.option = option
chart


# ##### VIZ Echarts Comparación de viajes mensual por intervalos 2 vs 3 años
# 
# * **Antes**: 2018 - 2019
# * **Durante**: 2020 - 2021
# * **Despues**: 2022 - 2023 - 2024
# 

# In[763]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, XAxis, YAxis, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Line

df_pivot_12421 = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)
df_pivot_12421['Mes'] = df_pivot_12421['Mes'].astype(str)

# Datos: meses 1-12 (string) para eje X
months = list(map(str, range(1, 13)))

# Agrupar por mes, sumando el total mensual
totales_mens_18_19 = df_pivot_12421[df_pivot_12421['Anio'].isin([2018, 2019])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_20_21 = df_pivot_12421[df_pivot_12421['Anio'].isin([2020, 2021])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()
totales_mens_22_23_24 = df_pivot_12421[df_pivot_12421['Anio'].isin([2022, 2023, 2024])].groupby([ 'Mes'])['Total'].sum().reindex(months, fill_value=np.nan).reset_index()

# Datos de cada año (por mes)
data_18_19 = totales_mens_18_19.Total.values
data_20_21 = totales_mens_20_21.Total.values
data_22_23_24 = totales_mens_22_23_24.Total.values

mark_area = {
    'data': [
        [{'xAxis': '4'}, {'xAxis': '6'}]  # Marca desde abril (4) a mayo (6)
    ],
    'itemStyle': {
        'color': 'rgba(255, 173, 177, 0.2)'
    }
}
chart = EChartsWidget()

xAxis = XAxis(type="category", data=months)
yAxis = YAxis(type="value")

colors = ['#80FFA5', '#FF0087', '#00DDFF']


line1 = Line(name="2018-2019", type="line", data=data_18_19, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None, markArea=mark_area)
line2 = Line(name="2020-2021", type="line", data=data_20_21, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)
line3 = Line(name="2022-2023-2024", type="line", data=data_22_23_24, smooth=True, showSymbol=True, lineStyle={'width': 2}, areaStyle=None)

option = Option()
option.title = Title(text="ETR: Comparación de viajes mensual por intervalos 2 vs 3 años")
option.tooltip = Tooltip(trigger="axis")
option.legend = Legend(data=["2018-2019", "2020-2021", "2022-2023-2024"], top='bottom')
option.xAxis = xAxis
option.yAxis = yAxis
option.color = colors
option.series = [line1, line2, line3]
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})


chart.option = option
chart


# ##### VIZ Echarts Donut Distribución total de viajes por periodo

# In[ ]:


import pandas as pd  # TODO
import numpy as np

df_pivot_12421.head()


# In[539]:


df_pivot_12421.info()


# In[540]:


etr_suma_2018_2019 = df_pivot_12421[df_pivot_12421['Anio'].isin([2018, 2019])]['Total'].sum()
etr_suma_2020_2021 = df_pivot_12421[df_pivot_12421['Anio'].isin([2020, 2021])]['Total'].sum()
etr_suma_2022_2023 = df_pivot_12421[df_pivot_12421['Anio'].isin([2022, 2023])]['Total'].sum()
etr_suma_2024_2025 = df_pivot_12421[df_pivot_12421['Anio'].isin([2024, 2025])]['Total'].sum()


# In[541]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Pie

data = [
    {"value": etr_suma_2018_2019, "name": "2018–2019"},
    {"value": etr_suma_2020_2021, "name": "2020–2021"},
    {"value": etr_suma_2022_2023, "name": "2022–2023"},
    {"value": etr_suma_2024_2025, "name": "2024–2025"}
]

# Colores opcionales
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

# Crear widget de gráfico
chart = EChartsWidget()

# Crear serie tipo "donut"
series = Pie(
    name="Viajes por periodo",
    type="pie",
    radius=["40%", "70%"],
    avoidLabelOverlap=False,
    itemStyle={
        "borderRadius": 10,
        "borderColor": "#fff",
        "borderWidth": 2
    },
    label={
        "show": True,
        "position": "inside",
        "formatter": "{d}%",
        "fontSize": 12
    },
    emphasis={
        "label": {
            "show": True,
            "fontSize": 20,
            "fontWeight": "bold"
        }
    },
    labelLine={"show": False},
    data=data
)

# Configurar opciones del gráfico
option = Option()
option.color = colors
option.title = Title(text="ETR: Distribución total de viajes por periodo", left="center")
option.tooltip = Tooltip(trigger="item", formatter="{b}: {c} ({d}%)")
option.legend = Legend(top="5%", left="center")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})
option.series = [series]

# Asignar opción al gráfico
chart.option = option
chart


# ##### VIZ Echarts Donut por Tipo de viaje por intervalos de dos años

# In[542]:


df_pivot_12421.head()


# In[543]:


df_pivot_12421.info()


# In[544]:


etr_negocio_2018_2019 = df_pivot_12421[df_pivot_12421['Anio'].isin([2018, 2019])]['Negocios y otros motivos profesionales'].sum()
etr_negocio_2020_2021 = df_pivot_12421[df_pivot_12421['Anio'].isin([2020, 2021])]['Negocios y otros motivos profesionales'].sum()
etr_negocio_2022_2023 = df_pivot_12421[df_pivot_12421['Anio'].isin([2022, 2023])]['Negocios y otros motivos profesionales'].sum()
etr_negocio_2024_2025 = df_pivot_12421[df_pivot_12421['Anio'].isin([2024, 2025])]['Negocios y otros motivos profesionales'].sum()

etr_ocio_2018_2019 = df_pivot_12421[df_pivot_12421['Anio'].isin([2018, 2019])]['Ocio, recreo y vacaciones'].sum()
etr_ocio_2020_2021 = df_pivot_12421[df_pivot_12421['Anio'].isin([2020, 2021])]['Ocio, recreo y vacaciones'].sum()
etr_ocio_2022_2023 = df_pivot_12421[df_pivot_12421['Anio'].isin([2022, 2023])]['Ocio, recreo y vacaciones'].sum()
etr_ocio_2024_2025 = df_pivot_12421[df_pivot_12421['Anio'].isin([2024, 2025])]['Ocio, recreo y vacaciones'].sum()

etr_otros_2018_2019 = df_pivot_12421[df_pivot_12421['Anio'].isin([2018, 2019])]['Otros motivos'].sum()
etr_otros_2020_2021 = df_pivot_12421[df_pivot_12421['Anio'].isin([2020, 2021])]['Otros motivos'].sum()
etr_otros_2022_2023 = df_pivot_12421[df_pivot_12421['Anio'].isin([2022, 2023])]['Otros motivos'].sum()
etr_otros_2024_2025 = df_pivot_12421[df_pivot_12421['Anio'].isin([2024, 2025])]['Otros motivos'].sum()


# In[545]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, Title, Tooltip, Legend, Toolbox, Grid
from ipecharts.option.series import Pie

# Datos ya procesados
datos_motivos = {
    "Negocio": [
        {"value": etr_negocio_2018_2019, "name": "2018–2019"},
        {"value": etr_negocio_2020_2021, "name": "2020–2021"},
        {"value": etr_negocio_2022_2023, "name": "2022–2023"},
        {"value": etr_negocio_2024_2025, "name": "2024–2025"}
    ],
    "Ocio": [
        {"value": etr_ocio_2018_2019, "name": "2018–2019"},
        {"value": etr_ocio_2020_2021, "name": "2020–2021"},
        {"value": etr_ocio_2022_2023, "name": "2022–2023"},
        {"value": etr_ocio_2024_2025, "name": "2024–2025"}
    ],
    "Otros": [
        {"value": etr_otros_2018_2019, "name": "2018–2019"},
        {"value": etr_otros_2020_2021, "name": "2020–2021"},
        {"value": etr_otros_2022_2023, "name": "2022–2023"},
        {"value": etr_otros_2024_2025, "name": "2024–2025"}
    ]
}

# Crear widget de gráfico
chart = EChartsWidget()

# Colores para los períodos
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

# Crear lista de series para cada gráfico
series = []
for idx, (motivo, data) in enumerate(datos_motivos.items()):
    series.append(
        Pie(
            name=motivo,
            type="pie",
            radius=["30%", "60%"],
            center=[f"{(idx + 1) * 25}%", "50%"],
            avoidLabelOverlap=False,
            itemStyle={
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2
            },
            label={
                "show": True,
                "position": "center",
                "formatter": motivo,
                "fontSize": 16,
                "fontWeight": "bold"
            },
            emphasis={
                "label": {
                    "show": True,
                    "fontSize": 16,
                    "fontWeight": "bold"
                }
            },
            labelLine={"show": False},
            data=data
        )
    )

centers = ["20%", "50%", "80%"]

# Configurar opciones del gráfico
option = Option()
option.title = Title(text="ETR: Distribución de viajes por motivo y período", left="center")
option.tooltip = Tooltip(trigger="item", formatter="{a}<br/>{b}: {c} ({d}%)")
option.legend = Legend(orient="vertical", left="left")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})
option.series = series
option.color = colors

# Asignar opción al gráfico
chart.option = option
chart


# In[546]:


from ipecharts import EChartsWidget
from ipecharts.option import Option, Title, Tooltip, Legend, Toolbox
from ipecharts.option.series import Pie

# Datos por motivo
datos_motivos = {
    "Negocio": [
        {"value": etr_negocio_2018_2019, "name": "2018–2019"},
        {"value": etr_negocio_2020_2021, "name": "2020–2021"},
        {"value": etr_negocio_2022_2023, "name": "2022–2023"},
        {"value": etr_negocio_2024_2025, "name": "2024–2025"}
    ],
    "Ocio": [
        {"value": etr_ocio_2018_2019, "name": "2018–2019"},
        {"value": etr_ocio_2020_2021, "name": "2020–2021"},
        {"value": etr_ocio_2022_2023, "name": "2022–2023"},
        {"value": etr_ocio_2024_2025, "name": "2024–2025"}
    ],
    "Otros": [
        {"value": otros_2018_2019, "name": "2018–2019"},
        {"value": otros_2020_2021, "name": "2020–2021"},
        {"value": otros_2022_2023, "name": "2022–2023"},
        {"value": otros_2024_2025, "name": "2024–2025"}
    ]
}

# Crear widget de gráfico
chart = EChartsWidget()

# Crear las series de tipo "donut"
series = []
centers = ["20%", "50%", "80%"]
motivos = list(datos_motivos.keys())
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']

for i, motivo in enumerate(motivos):
    series.append(
        Pie(
            name=motivo,
            type="pie",
            radius=["30%", "60%"],
            center=[centers[i], "60%"],
            avoidLabelOverlap=False,
            itemStyle={
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2
            },
            label={
                "show": True,
                "position": "inside",
                "formatter": "{d}%",
                "fontSize": 12
            },
            emphasis={
                "label": {
                    "show": True,
                    "fontSize": 14,
                    "fontWeight": "bold"
                }
            },
            labelLine={"show": True},
            data=datos_motivos[motivo]
        )
    )

# Configurar opciones del gráfico
option = Option()
option.title = Title(text="ETR: Distribución de viajes por motivo y período", left="center", top="5%")
option.tooltip = Tooltip(trigger="item", formatter="{a}<br/>{b}: {c} ({d}%)")
option.legend = Legend(orient="vertical", left="left")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})
option.series = series
option.color = colors

# Asignar opción al gráfico
chart.option = option
chart


# In[547]:


import pandas as pd
from ipecharts import EChartsWidget
from ipecharts.option import Option, Tooltip, Legend, Grid, XAxis, YAxis, Title
from ipecharts.option.series import Bar

# Agrupa por año y suma cada motivo
df_grouped = df_pivot_12421.groupby('Anio').agg({
    'Negocios y otros motivos profesionales': 'sum',
    'Ocio, recreo y vacaciones': 'sum',
    'Otros motivos': 'sum'
}).reset_index()

anios = df_grouped['Anio'].astype(str).tolist()
negocio = df_grouped['Negocios y otros motivos profesionales'].tolist()
ocio = df_grouped['Ocio, recreo y vacaciones'].tolist()
otros = df_grouped['Otros motivos'].tolist()

series = [
    Bar(name='Negocio', data=negocio, stack='total', label={'show': False}),
    Bar(name='Ocio', data=ocio, stack='total', label={'show': False}),
    Bar(name='Otros', data=otros, stack='total', label={'show': False}),
]

option = Option()
option.tooltip = Tooltip(trigger='axis', axisPointer={'type': 'shadow'})
option.legend = Legend(orient="horizontal", top="bottom")
option.grid = Grid(left='3%', right='4%', bottom='3%', containLabel=True)
option.xAxis = XAxis(type='value')
option.yAxis = YAxis(type='category', data=anios)
option.series = series
option.title = Title(text="ETR: Totales apilados por motivo de viaje y año", left="center")
option.toolbox = Toolbox(show=True, feature={"saveAsImage": {"show": True}})

chart = EChartsWidget()
chart.option = option
chart


# #### 14297 [Viajes por principales características de los viajes](https://www.ine.es/jaxiT3/Tabla.htm?t=14297)

# La idea inicial era descargar el `csv` segmentado por destinos, motivo principal de los viajes mensuales, pero ha resultado imposible. He probado diferente combinaciones de años, pero la tupla 2020 - 2021 al tener muchos ceros no permite descargarlos. Y en la descarga del dataset completo da error (_No existen ficheros de descarga disponibles en estos momentos. Inténtelo de nuevo más tarde_). Mi gozo en un pozo. 🐰🥚

# Asi que me quedo solamente con los datos segmentados por _destinos_.

# ![image.png](attachment:image.png)

# In[ ]:


import pandas as pd

df_14297 = pd.read_csv('./data/14297_x_destinos.csv', sep=';', encoding='latin-1')

# Eliminar las cols con valores únicos
unique_counts = df_14297.describe(include='all').loc['unique']
cols_v_unique = unique_counts[unique_counts == 1].index.tolist()

df_14297_filtrado = df_14297.drop(columns=cols_v_unique)

# Replace los separadores de miles '.' de la columna Total
df_14297_filtrado.Total = df_14297_filtrado.Total.replace('\.', '', regex=True)

# Elimino los registros 'Total'
df_14297_filtrado = df_14297_filtrado[df_14297_filtrado['Destino principal'] != 'Total']
df_14297_filtrado.drop('Alojamiento principal: Nivel 2', axis=1)

# Convert to number
df_14297_filtrado.Total = df_14297_filtrado.Total.apply((pd.to_numeric), errors='coerce')

df_filtrado_14297_pivot_dest = df_14297_filtrado.pivot_table(
    index='Periodo',
    columns='Destino principal',
    values='Total',
    aggfunc='sum'
).sort_index()

df_filtrado_14297_pivot_dest['Mes'] = df_filtrado_14297_pivot_dest.index.str[-2:]
df_filtrado_14297_pivot_dest['Anio'] = df_filtrado_14297_pivot_dest.index.str[:4]

# * Convertimos en ``int`` tmb el año y el mes
col_no = ['Anio', 'Mes']
df_filtrado_14297_pivot_dest[col_no] = df_filtrado_14297_pivot_dest[col_no].apply(pd.to_numeric, errors='coerce')

df_filtrado_14297_pivot_dest.to_csv('./data/df_filtrado_14297_pivot_dest.csv')

display('df_filtrado_14297_pivot_dest.head()', 'df_filtrado_14297_pivot_dest.tail()')


# ### Webscrapping festivos España
# 
# [Ver en detalle](./webScrapping.ipynb)
# 
# [Calendarios laborales](https://www.calendarioslaborales.com)

# ![img](./img/calendarios.PNG)

# #### Calendario_festivos_15_25.csv

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_calendar = pd.read_csv("./data/calendario_festivos_15_25.csv", index_col=0)
display('df_calendar.head()', 'df_calendar.tail()')


# In[ ]:


df_calendar.info()


# In[ ]:


df_calendar[df_calendar.festivo.isna()]


# Los _missinngs_ resultan ser el día de "Día de Todos los Santos" en 2019 para todas las provincias. Actualizo campo y csv.

# In[ ]:


df_calendar.loc[df_calendar.festivo.isna(), 'festivo'] = 'Día de Todos los Santos'
df_calendar.info()


# In[ ]:


df_calendar.to_csv('./data/calendario_festivos_15_25.csv')


# Índice de concentración festiva

# In[ ]:


# df_concentracion_festiva = 
df_calendar.groupby(['provincia', 'anio', 'mes'])['festivo'].count()


# In[ ]:


df_concentracion_festiva = df_calendar.groupby(['provincia', 'anio', 'mes'])['festivo'].count()


# In[ ]:


df_concentracion_festiva.loc['Alava'][2019]


# ## H1 Se viaja más despúes de la _pandemia_? Viajes antes y después de la pandemia.
#     - En España, el número de viajes ha aumentado tras la pandemia (2020).
# 

# In[980]:


df_h1_fron = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_h1_ert = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)

df_pivot_12421 = df_h1_fron.rename(columns={'Negocios y otros motivos profesionales': 'Negocio',
                                'Total': 'Total_ETR'})

df_pivot_13864 = df_h1_fron.rename(columns={'Negocio, motivos profesionales': 'Negocio',
                                        'Total': 'Total_FRONTUR'})

df_merged_h1 = pd.merge(df_pivot_12421[['Anio', 'Mes', 'Total_ETR']],
                     df_pivot_13864[['Anio', 'Mes', 'Total_FRONTUR']],
                     on=['Anio', 'Mes'],
                     how='outer').sort_values(by=['Anio', 'Mes'])

df_merged_h1['Total'] = df_merged_h1['Total_ETR'] + df_merged_h1['Total_FRONTUR']

df_merged_h1['Total_ETR'] = pd.to_numeric(df_merged_h1['Total_ETR'], errors='coerce')
df_merged_h1['Total_FRONTUR'] = pd.to_numeric(df_merged_h1['Total_FRONTUR'], errors='coerce')
df_merged_h1['Total']  = df_merged_h1[['Total_ETR', 'Total_FRONTUR']].sum(axis=1, skipna=True)
df_merged_h1.tail(10)
df_merged_h1.info()


# ##### VIZ Evolución viajes 2015 - 2025: FRONTUR + ETR

# In[1055]:


df_merged_h1


# In[1056]:


import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que las columnas estén correctamente nombradas
df_merged_h1['anio'] = pd.to_numeric(df_merged_h1['anio'], errors='coerce')
df_merged_h1['Total'] = pd.to_numeric(df_merged_h1['Total'], errors='coerce')

# Agrupar y sumar total anual
total_anual = df_merged_h1.groupby('anio')['Total'].sum().reset_index()
colors = [
    '#80FFA5', '#00DDFF', "#08090A", '#FF0087', '#FFBF00',
    '#7FFF00', '#1E90FF', '#FF4500', '#FFD700', '#00FA9A', '#BA55D3'
]

# Crear gráfico
plt.figure(figsize=(10, 6))
bars = plt.bar(total_anual['anio'], total_anual['Total'], color=colors, alpha=0.4)

# Añadir etiquetas en cada barra
for bar, anio in zip(bars, total_anual['anio']):
    height = bar.get_height()
    plt.annotate(f'{int(height):,}',  # con separadores de miles
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # desplazamiento hacia arriba
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)
    if anio == 2020:
        bar.set_edgecolor('red')
        bar.set_linewidth(3)
    else:
        bar.set_edgecolor('none')

# Personalización
plt.title('Evolución viajes 2015 - 2025. FRONTUR + ETR', fontsize=14)
plt.xlabel('Año')
plt.ylabel('Total de viajes')
plt.grid(False)
plt.xticks(total_anual['anio'])
plt.tight_layout()
plt.savefig('./img/viz-evol_viajes-etr-fro.png')
plt.show()


# ##### VIZ Comparación anual de viajes: FRONTUR vs ETR 2015 -2025

# In[1057]:


import pandas as pd
import matplotlib.pyplot as plt

# Asegurar tipos numéricos
df_h1_fron['Anio'] = pd.to_numeric(df_h1_fron['Anio'], errors='coerce')
df_h1_fron['Total'] = pd.to_numeric(df_h1_fron['Total'], errors='coerce')
df_h1_ert['Anio'] = pd.to_numeric(df_h1_ert['Anio'], errors='coerce')
df_h1_ert['Total'] = pd.to_numeric(df_h1_ert['Total'], errors='coerce')

# Agrupar por año
total_fron = df_h1_fron.groupby('Anio')['Total'].sum()
total_ert = df_h1_ert.groupby('Anio')['Total'].sum()

# Combinar en un solo DataFrame
df_comparado = pd.DataFrame({
    'FRONTUR': total_fron,
    'ETR': total_ert
}).reset_index()

# --------------------
# Gráfico de barras
# --------------------
import numpy as np

x = np.arange(len(df_comparado['Anio']))  # Posiciones de los grupos
width = 0.35  # Ancho de las barras

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, df_comparado['FRONTUR'], width, label='FRONTUR', color='#00DDFF')
plt.bar(x + width/2, df_comparado['ETR'], width, label='ETR', color='#f0a8a8')

# Etiquetas y leyenda
plt.xlabel('Año')
plt.ylabel('Total de viajes')
plt.title('Comparación anual de viajes: FRONTUR vs ETR')
plt.xticks(x, df_comparado['Anio'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.grid(False)
plt.savefig('./img/viz-comp_viajes-etr-fro.png')
plt.show()


# ##### VIZ Comparación anual de viajes: FRONTUR vs ETR y Total Combinado

# In[1058]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Asegurar tipos
df_h1_fron['Anio'] = pd.to_numeric(df_h1_fron['Anio'], errors='coerce')
df_h1_fron['Total'] = pd.to_numeric(df_h1_fron['Total'], errors='coerce')
df_h1_ert['Anio'] = pd.to_numeric(df_h1_ert['Anio'], errors='coerce')
df_h1_ert['Total'] = pd.to_numeric(df_h1_ert['Total'], errors='coerce')

# Agrupación anual
total_fron = df_h1_fron.groupby('Anio')['Total'].sum()
total_ert = df_h1_ert.groupby('Anio')['Total'].sum()

# Combinar datos
df_comparado = pd.DataFrame({
    'FRONTUR': total_fron,
    'ETR': total_ert
})
df_comparado['TOTAL'] = df_comparado['FRONTUR'] + df_comparado['ETR']
df_comparado = df_comparado.reset_index()

# --------------------
# Gráfico de barras agrupadas
# --------------------
x = np.arange(len(df_comparado['Anio']))
width = 0.25

plt.figure(figsize=(14, 7))
plt.bar(x - width, df_comparado['FRONTUR'], width, label='FRONTUR', color='#00DDFF')
plt.bar(x, df_comparado['ETR'], width, label='ETR', color='#f0a8a8')
plt.bar(x + width, df_comparado['TOTAL'], width, label='TOTAL', color='#A0A0A0')

# Etiquetas
plt.xlabel('Año')
plt.ylabel('Total de viajes')
plt.title('Comparación anual de viajes: FRONTUR vs ETR y Total Combinado')
plt.xticks(x, df_comparado['Anio'])
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('./img/viz-comp-viajes-etr-fro-total.png')
plt.show()


# In[1009]:


df_merged_h1.info()


# In[1061]:


import numpy as np

months = list(range(1, 13))

t_mens_18_19 = df_merged_h1[df_merged_h1['anio'].isin([2018, 2019])].groupby('mes')['Total'].sum().reindex(months, fill_value=0).reset_index()
t_mens_20_21 = df_merged_h1[df_merged_h1['anio'].isin([2020, 2021])].groupby('mes')['Total'].sum().reindex(months, fill_value=0).reset_index()
t_mens_22_23 = df_merged_h1[df_merged_h1['anio'].isin([2022, 2023])].groupby('mes')['Total'].sum().reindex(months, fill_value=0).reset_index()
t_mens_24_25 = df_merged_h1[df_merged_h1['anio'].isin([2024, 2025])].groupby('mes')['Total'].sum().reindex(months, fill_value=0).reset_index()

d_18_19 = t_mens_18_19.Total.values
d_20_21 = t_mens_20_21.Total.values
d_22_23 = t_mens_22_23.Total.values
d_24_25 = t_mens_24_25.Total.values


# In[1062]:


import matplotlib.pyplot as plt

# Lista de nombres para cada grupo de años
labels = ['2018–2019', '2020–2021', '2022–2023', '2024–2025']
data = [d_18_19, d_20_21, d_22_23, d_24_25]
colors = ['#80FFA5', '#FF0087', '#00DDFF', '#37A2FF']


plt.figure(figsize=(14, 8))

# Graficar cada serie
for i, y in enumerate(data):
    plt.plot(range(1, 13), y, label=labels[i], color=colors[i], marker='o')

plt.title('Total viajes por mes en intervalos de dos años')
plt.xlabel('Mes')
plt.ylabel('Total viajes')
plt.xticks(range(1, 13))
plt.axvspan(4, 6, color='red', alpha=0.1)
plt.legend(title='Periodos', loc='upper right')
plt.grid(False)
plt.tight_layout()
plt.savefig('./img/viz-evol_viajes-etr-fro-mens-bi.png')
plt.show()


# ## H2 Se viaja más por ocio que por trabajo? Motivos de viaje: _turismo_ vs _negocios_
#     - El turismo nacional de ocio ha aumentado más que los viajes por otros motivos (trabajo, estudios, etc.) «Motivo del viaje»

# In[616]:


df_h2_12421 = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)
df_h2_12421.head()
df_h2_13864 = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
df_h2_13864.head()
df_h2_merged = pd.merge(
    df_h2_12421,
    df_h2_13864,
    on=['Anio', 'Mes'],
    how='outer',
    suffixes=('_12421', '_13864')
).sort_values(by=['Anio', 'Mes'])

df_h2_merged.head()

df_h2_merged['Negocio'] = df_h2_merged[['Negocios y otros motivos profesionales', 'Negocio, motivos profesionales']].sum(axis=1, skipna=True)
df_h2_merged['Ocio'] = df_h2_merged[['Ocio, recreo y vacaciones_12421', 'Ocio, recreo y vacaciones_13864']].sum(axis=1, skipna=True)

df_h2_merged



# ##### Distribución total de viajes: Negocio vs. Ocio

# In[1063]:


import matplotlib.pyplot as plt

# Suma total de cada tipo
total_negocio = df_h2_merged['Negocio'].sum()
total_ocio = df_h2_merged['Ocio'].sum()

# Datos y etiquetas
labels = ['Negocio', 'Ocio']
sizes = [total_negocio, total_ocio]
colors = ["#9BE5B0", "#D9A0E8"]

# Crear el gráfico
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribución total de viajes: Negocio vs. Ocio')
plt.axis('equal')  # Para que sea un círculo
plt.savefig('./img/viz-distr-negocio-ocio.png')
plt.show()


# ##### Comparación anual de viajes por Negocio y Ocio

# In[1064]:


import matplotlib.pyplot as plt
import numpy as np

# Obtener años únicos ordenados
anios = sorted(df_h2_merged['Anio'].unique())

# Calcular sumas anuales por categoría
negocio = [df_h2_merged[df_h2_merged['Anio'] == year]['Negocio'].sum() for year in anios]
ocio = [df_h2_merged[df_h2_merged['Anio'] == year]['Ocio'].sum() for year in anios]

x = np.arange(len(anios))
width = 0.35

plt.figure(figsize=(12, 7))

# Ejemplo: marcar confinamiento (por ejemplo en 2020)
if 2020 in anios:
    idx_2020 = anios.index(2020)
    plt.axvspan(idx_2020 - 0.5, idx_2020 + 0.5, color='red', alpha=0.1)

bars1 = plt.barh(x - width/2, negocio, width, label='Negocio', color="#9BE5B0")
bars2 = plt.barh(x + width/2, ocio, width, label='Ocio', color='#D9A0E8')

# Añadir etiquetas con formato legible
for bars in [bars1, bars2]:
    for bar in bars:
        width_bar = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        plt.text(width_bar + max(negocio + ocio)*0.01, y_pos, f'{int(width_bar):,}', 
                 va='center', fontsize=10)

plt.yticks(x, anios)
plt.xlabel('Total de viajes')
plt.title('Comparación anual de viajes por Negocio y Ocio')
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('./img/viz-comp-viajes-neg-ocio.png')
plt.show()


# ##### VIZ Comparación de viajes por Negocio y Ocio (periodos bianuales)

# In[1065]:


import matplotlib.pyplot as plt
import numpy as np

# Crear DataFrame resumen con comprensión para evitar repeticiones
periodos = [(2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025)]
labels = ['2018-19', '2020-21', '2022-23', '2024-25']

negocio = [df_h2_merged[df_h2_merged['Anio'].isin(years)]['Negocio'].sum() for years in periodos]
ocio = [df_h2_merged[df_h2_merged['Anio'].isin(years)]['Ocio'].sum() for years in periodos]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))

# Marcar el área de confinamiento si aplica a alguna barra (ejemplo con el segundo grupo)
plt.axvspan(0.5, 1.5, color='red', alpha=0.1)

bars1 = plt.barh(x - width/2, negocio, width, label='Negocio', color="#9BE5B0")
bars2 = plt.barh(x + width/2, ocio, width, label='Ocio', color='#D9A0E8')

# Añadir etiquetas con formato legible
for bars in [bars1, bars2]:
    for bar in bars:
        width_bar = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        plt.text(width_bar + max(negocio + ocio)*0.01, y_pos, f'{int(width_bar):,}', 
                 va='center', fontsize=10)

plt.yticks(x, labels)
plt.xlabel('Total de viajes')
plt.title('Comparación de viajes por Negocio y Ocio (periodos bianuales)')
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('./img/viz-comp-viajes-neg-ocio-bi.png')
plt.show()


# ## H3 El turismo internacional se ha recuperado más rápidamente que el nacional.
#     - Comprobar si los turistas extranjeros regresaron en mayor número o antes que los residentes nacionales.
#     - Comparar la evolución año a año desde 2020.

# In[979]:


import pandas as pd
h3_df_fron = pd.read_csv('./data/frt_13864_clean.csv', index_col=0)
h3_df_ert = pd.read_csv('./data/ert_12421_clean.csv', index_col=0)


# ##### VIZ Evolución viajes internacionales vs. nacionales (mes, años 2015 -2025)

# In[1066]:


import pandas as pd
import matplotlib.pyplot as plt

# Suponiendo que ya tienes df_fron y df_ert cargados
# Seleccionar columnas relevantes y renombrar
df_fron_total = h3_df_fron[['Anio', 'Mes', 'Total']].copy()
df_ert_total = h3_df_ert[['Anio', 'Mes', 'Total']].copy()

df_fron_total.rename(columns={'Total': 'Total_FRONTUR'}, inplace=True)
df_ert_total.rename(columns={'Total': 'Total_ETR'}, inplace=True)

# Unir los dataframes por Año y Mes
df_merged = pd.merge(df_fron_total, df_ert_total, on=['Anio', 'Mes'], how='outer').sort_values(by=['Anio', 'Mes'])

# Crear columna Periodo tipo fecha
df_merged['Periodo'] = pd.to_datetime(df_merged['Anio'].astype(str) + '-' + df_merged['Mes'].astype(str).str.zfill(2))

plt.figure(figsize=(12, 6))

plt.plot(df_merged['Periodo'], df_merged['Total_FRONTUR'], label='FRONTUR', color='#00BFFF')
plt.plot(df_merged['Periodo'], df_merged['Total_ETR'], label='ETR', color='#FF0087')

plt.title('Evolución mensual de viajeros: FRONTUR vs ETR')
plt.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'), color='red', alpha=0.1)
plt.xlabel('Periodo')
plt.ylabel('Viajeros totales')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.grid(False)
plt.tight_layout()
plt.savefig('./img/viz-evol_viajes-inac-nac.png')
plt.show()


# ## H4 Influyen los festivos en el número de viajes? Las estaciones (tendencias _estacionales_)?
# 

# ##### VIZ multi bar + line 2020 - 2024

# In[1052]:


import pandas as pd
df_merged_h1.head(10)


# In[1053]:


df_merged_h1 = df_merged_h1.rename(columns={'Anio':'anio', 'Mes':'mes'})
df_merged_h1


# In[ ]:


df_merged_h1['anio'] = pd.to_numeric(df_merged_h1['anio'], errors='coerce')
df_merged_h1['mes'] = pd.to_numeric(df_merged_h1['mes'], errors='coerce')


# In[800]:


h6_calendario_fest_15_25 = pd.read_csv('./data/calendario_festivos_15_25.csv', index_col=0)


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------- Cargar datos ---------
df_viajeros = df_merged_h1 #pd.read_csv('viajeros.csv')  # columnas: Anio, Mes, Total
df_festivos = h6_calendario_fest_15_25 #pd.read_csv('festivos.csv')

# --------- Preprocesamiento ---------
# Convertir nombre de mes a número
meses = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}
df_festivos['mes_num'] = df_festivos['mes'].map(meses)

# Contar festivos por mes y año
festivos_por_mes = df_festivos.groupby(['anio', 'mes']).size().reset_index(name='num_festivos')

# Renombrar columnas para hacer merge
df_viajeros.rename(columns={'Anio': 'anio', 'Mes': 'mes'}, inplace=True)

# Combinar datasets
df = pd.merge(df_viajeros, festivos_por_mes, how='left', on=['anio', 'mes'])
df['num_festivos'] = df['num_festivos'].fillna(0)

# Limitar a los años deseados
df = df[df['anio'].isin([2020, 2021, 2022, 2023, 2024])]   # 2025
df = df.sort_values(by=['anio', 'mes'])

# --------- Crear subgráficas ---------
anios = [2020, 2021, 2022, 2023, 2024]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharey=True)
axs = axs.flatten()

for i, anio in enumerate(anios):
    ax1 = axs[i]
    df_year = df[df['anio'] == anio]

    # Eje X como mes numérico
    x = df_year['mes']
    y1 = df_year['Total']
    y2 = df_year['num_festivos']

    # Línea de viajeros
    ax1.plot(x, y1, marker='o', color='#FFBF00', label='Total Viajeros')
    ax1.set_title(f'Año {anio}')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    ax1.set_ylabel('Viajeros')
    ax1.grid(False)

    # Eje secundario para festivos
    ax2 = ax1.twinx()
    ax2.bar(x, y2, width=0.5, alpha=0.1, color='red', label='Festivos')
    ax2.set_ylabel('Festivos')
    ax2.grid(False)

    # Solo mostrar leyenda en la primera subgráfica
    if i == 0:
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

# Oculta el último subplot si hay uno vacío
if len(anios) < len(axs):
    axs[-1].axis('off')

# Ajustar diseño general
plt.suptitle('Evolución mensual de viajeros en España y concentración festiva (2020–2024)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./img/evol_concentrac_festiva_es.png')
plt.show()


# ##### VIZ multi bar + line 2015 - 2020

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------- Cargar datos ---------
df_viajeros = df_merged_h1 #pd.read_csv('viajeros.csv')  # columnas: Anio, Mes, Total
df_festivos = h6_calendario_fest_15_25 #pd.read_csv('festivos.csv')

# --------- Preprocesamiento ---------
# Convertir nombre de mes a número
meses = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}
df_festivos['mes_num'] = df_festivos['mes'].map(meses)

# Contar festivos por mes y año
festivos_por_mes = df_festivos.groupby(['anio', 'mes']).size().reset_index(name='num_festivos')

# Renombrar columnas para hacer merge
df_viajeros.rename(columns={'Anio': 'anio', 'Mes': 'mes'}, inplace=True)

# Combinar datasets
df = pd.merge(df_viajeros, festivos_por_mes, how='left', on=['anio', 'mes'])
df['num_festivos'] = df['num_festivos'].fillna(0)

# Limitar a los años deseados
df = df[df['anio'].isin([2015, 2016, 2017, 2018, 2019])]   # 2025
df = df.sort_values(by=['anio', 'mes'])

# --------- Crear subgráficas ---------
anios = [2015, 2016, 2017, 2018, 2019]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharey=True)
axs = axs.flatten()

for i, anio in enumerate(anios):
    ax1 = axs[i]
    df_year = df[df['anio'] == anio]

    # Eje X como mes numérico
    x = df_year['mes']
    y1 = df_year['Total']
    y2 = df_year['num_festivos']

    # Línea de viajeros
    ax1.plot(x, y1, marker='o', color='#FFBF00', label='Total Viajeros')
    ax1.set_title(f'Año {anio}')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    ax1.set_ylabel('Viajeros')
    ax1.grid(False)

    # Eje secundario para festivos
    ax2 = ax1.twinx()
    ax2.bar(x, y2, width=0.5, alpha=0.1, color='red', label='Festivos')
    ax2.set_ylabel('Festivos')
    ax2.grid(False)

    # Solo mostrar leyenda en la primera subgráfica
    if i == 0:
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

# Oculta el último subplot si hay uno vacío
if len(anios) < len(axs):
    axs[-1].axis('off')

# Ajustar diseño general
plt.suptitle('Evolución mensual de viajeros en España y concentración festiva (2015-2019)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./img/evol_concentrac_festiva_es.png')
plt.show()


# ## H5 Los españoles viajan más dentro de España o prefieren el extranejo después de la pandemia?

# In[1038]:


import pandas as pd

h5_df_14297 = pd.read_csv('./data/df_filtrado_14297_pivot_dest.csv', index_col=0)
h5_df_14297 = h5_df_14297.reset_index()
h5_df_14297 = h5_df_14297[['Periodo', 'España', 'Extranjera']]
h5_df_14297.head()


# ##### VIZ Evolución temporal de viajes: España vs Extranjero

# In[1067]:


import pandas as pd
import matplotlib.pyplot as plt

h5_df_14297['fecha'] = pd.to_datetime(h5_df_14297['Periodo'], format='%YM%m')

plt.figure(figsize=(12, 6))

plt.plot(h5_df_14297['fecha'], h5_df_14297['España'], label='España', color='#1f77b4')
plt.plot(h5_df_14297['fecha'], h5_df_14297['Extranjera'], label='Extranjera', color='#ff7f0e')

plt.title('Evolución temporal de viajes: España vs Extranjero')
plt.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'), color='red', alpha=0.1)
plt.xlabel('Fecha')
plt.ylabel('Número de viajes')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('./img/viz-evol_viajes-esp-extranj.png')
plt.show()


# ##### VIZ Viajes totales agrupados por periodo: España vs Extranjero

# In[1048]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Suponiendo que h5_df_14297 es tu DataFrame original
h5_df_14297['fecha'] = pd.to_datetime(h5_df_14297['Periodo'], format='%YM%m')
h5_df_14297['anio'] = h5_df_14297['fecha'].dt.year

# Definir función para asignar periodo
def asignar_periodo(anio):
    if 2018 <= anio <= 2019:
        return '2018-2019'
    elif 2020 <= anio <= 2021:
        return '2020-2021'
    elif 2022 <= anio <= 2024:
        return '2022-2024'
    else:
        return 'Otros'

h5_df_14297['periodo'] = h5_df_14297['anio'].apply(asignar_periodo)

# Agrupar sumando viajes por periodo y tipo
resumen = h5_df_14297.groupby('periodo')[['España', 'Extranjera']].sum().reindex(['2018-2019', '2020-2021', '2022-2024'])

# Preparar gráfico
x = np.arange(len(resumen))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, resumen['España'], width, label='España', color='#1f77b4', alpha=0.5)
plt.bar(x + width/2, resumen['Extranjera'], width, label='Extranjero', color='#ff7f0e', alpha=0.5)

plt.xticks(x, resumen.index)
plt.xlabel('Periodo')
plt.ylabel('Número de viajes')
plt.title('Viajes totales agrupados por periodo: España vs Extranjero')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()



# ## H6 Tras la pandemia, los destinos costeros se han convertido en la opción preferida.
#     - Analizamos por comunidad autónoma en ETR y/o FRONTUR si hay más viajes hacia zonas como Andalucía, Comunidad Valenciana, Canarias, Islas Baleares, etc.

# * FRONTUR: df_13884_pivot_ccaa_total 
# * ETR: df_filtrado_14297_pivot_dest
# * ccaa_de_costa []
# * Total: FRO + ETR

# ##### VIZ FRONTUR CCAA

# In[888]:


df_13884_pivot_ccaa_total.head(1)


# In[829]:


df_13884_pivot_ccaa_total.info()


# In[1068]:


import pandas as pd
import matplotlib.pyplot as plt

# 1. Limpiar nombres de columnas y del índice
df_viajeros = pd.read_csv('./data/13884_15_25_ccaa.csv', index_col=0)
df_viajeros.columns = df_viajeros.columns.str.replace(r'^\d{2} ', '', regex=True).str.strip().str.lower()

ccaa_con_costa = [ccaa.lower() for ccaa in ccaa_con_costa]

# 3. Sumar viajeros por comunidad
viajeros_totales = df_viajeros.sum()
viajeros_totales.index = viajeros_totales.index.str.strip().str.lower()

# 4. Agrupar por tipo
total_con_costa = viajeros_totales[viajeros_totales.index.isin(ccaa_con_costa)].sum()
total_sin_costa = viajeros_totales[~viajeros_totales.index.isin(ccaa_con_costa)].sum()

# 5. Crear serie para gráfico
serie_donut = pd.Series({'Con costa': total_con_costa, 'Sin costa': total_sin_costa})

# 6. Graficar donut
colors = ["#00DDFF", '#FFBF00']

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    serie_donut,
    labels=serie_donut.index,
    autopct='%.1f%%',
    pctdistance=0.75,
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4)
)

ax.set_title('FRO: Distribución total de viajeros: CCAA con costa vs sin costa')
plt.tight_layout()
plt.savefig('./img/viz-distr-costa-no-costa.png')
plt.show()


# In[1069]:


import pandas as pd
import matplotlib.pyplot as plt

# ----------- 1. Preparar DataFrame -----------
df_viajero_barh_stacked = df_13884_pivot_ccaa_total.copy()

# Normalizar nombres de columnas: quitar prefijos "01 " y pasar a minúsculas
df_viajero_barh_stacked.columns = (
    df_viajero_barh_stacked.columns
    .str.replace(r'^\d{2} ', '', regex=True)
    .str.strip()
    .str.lower()
)

# Convertir índice a fecha y extraer año
df_viajero_barh_stacked.index = pd.to_datetime(df_viajero_barh_stacked.index, format='%YM%m')
df_viajero_barh_stacked['anio'] = df_viajero_barh_stacked.index.year

ccaa_con_costa_lower = [ccaa.lower() for ccaa in ccaa_con_costa]

# ----------- 3. Agrupación y separación -----------

# Sumar viajeros anualmente
df_sum_anual = df_viajero_barh_stacked.groupby('anio').sum(numeric_only=True)

# Verificar qué columnas de con_costa realmente existen (por si hay algún error tipográfico)
ccaa_con_costa_existentes = [col for col in ccaa_con_costa_lower if col in df_sum_anual.columns]
ccaa_sin_costa = [col for col in df_sum_anual.columns if col not in ccaa_con_costa_existentes]

# Añadir columnas agrupadas
df_sum_anual['con costa'] = df_sum_anual[ccaa_con_costa_existentes].sum(axis=1)
df_sum_anual['sin costa'] = df_sum_anual[ccaa_sin_costa].sum(axis=1)

# Extraer solo columnas necesarias
df_stacked = df_sum_anual[['con costa', 'sin costa']]

# ----------- 4. Graficar -----------

colors = ['#00DDFF', '#FFBF00']  # con costa, sin costa

ax = df_stacked.plot(
    kind='barh',
    stacked=True,
    figsize=(10, 6),
    color=colors
)

ax.set_xlabel('Viajeros totales')
ax.set_ylabel('Año')
ax.set_title('FRO: Viajeros totales por año — CCAA con costa vs sin costa')
ax.legend(loc='lower right')
ax.grid(False)
plt.tight_layout()
plt.savefig('./img/viz-bar-costa-no-costa.png')
plt.show()


# In[864]:


df_filtrado_14297_pivot_dest


# In[865]:


df_filtrado_14297_pivot_dest.info()


# ##### VIZ ETR CCAA

# In[1070]:


import pandas as pd
import matplotlib.pyplot as plt

# df_viajeros_etr =df_filtrado_14297_pivot_dest
df_viajeros_etr = pd.read_csv('./data/df_filtrado_14297_pivot_dest.csv', index_col=0)

# Se limpian los nombres de las columnas (extraer solo el nombre de la comunidad)
# Extraer solo el nombre de comunidad (remover el código tipo "01 ")
df_viajeros_etr.columns = df_viajeros_etr.columns.str.replace(r'^\d{2} ', '', regex=True).str.strip().str.lower()
ccaa_con_costa = [ccaa.lower() for ccaa in ccaa_con_costa]

# Sumar viajeros por comunidad (a lo largo de todos los meses)
viajeros_totales = df_viajeros_etr.sum()
viajeros_totales.index = viajeros_totales.index.str.strip().str.lower()

# Agrupar por tipo
total_con_costa = viajeros_totales[viajeros_totales.index.isin(ccaa_con_costa)].sum()
total_sin_costa = viajeros_totales[~viajeros_totales.index.isin(ccaa_con_costa)].sum()


# Crear serie resumen
serie_donut = pd.Series(
    {'Con costa': total_con_costa, 'Sin costa': total_sin_costa}
)

colors = ["#f0a8a8", '#FFBF00']

fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    serie_donut,
    labels=serie_donut.index,
    autopct='%.1f%%',
    # autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*serie_donut.sum()):,})",
    pctdistance=0.75,
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4)
)
# autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*serie_donut.sum()):,})",

ax.set_title('ETR: Distribución total de viajeros: CCAA con costa vs sin costa')
plt.tight_layout()
plt.savefig('./img/viz-distr_viajes-costa-no-costa-etr.png')
plt.show()


# In[1071]:


import pandas as pd
import matplotlib.pyplot as plt

# ----------- 1. Preparar DataFrame -----------
# df_viajero_barh_stacked = df_filtrado_14297_pivot_dest.copy()
df_viajero_barh_stacked = pd.read_csv('./data/df_filtrado_14297_pivot_dest.csv', index_col=0)


# Limpiar nombres de columnas: quitar prefijos "01 " y pasar a minúsculas
df_viajero_barh_stacked.columns = (
    df_viajero_barh_stacked.columns
    .str.replace(r'^\d{2} ', '', regex=True)
    .str.strip()
    .str.lower()
)

# Convertir índice a fecha y extraer año
df_viajero_barh_stacked.index = pd.to_datetime(df_viajero_barh_stacked.index, format='%YM%m')
df_viajero_barh_stacked['anio'] = df_viajero_barh_stacked.index.year

ccaa_con_costa_lower = [ccaa.lower() for ccaa in ccaa_con_costa]

# ----------- 3. Agrupación y separación -----------

# Sumar viajeros anualmente
df_sum_anual = df_viajero_barh_stacked.groupby('anio').sum(numeric_only=True)

# Verificar qué columnas de con_costa realmente existen (por si hay algún error tipográfico)
ccaa_con_costa_existentes = [col for col in ccaa_con_costa_lower if col in df_sum_anual.columns]
ccaa_sin_costa = [col for col in df_sum_anual.columns if col not in ccaa_con_costa_existentes]

# Añadir columnas agrupadas
df_sum_anual['con costa'] = df_sum_anual[ccaa_con_costa_existentes].sum(axis=1)
df_sum_anual['sin costa'] = df_sum_anual[ccaa_sin_costa].sum(axis=1)

# Extraer solo columnas necesarias
df_stacked = df_sum_anual[['con costa', 'sin costa']]

# ----------- 4. Graficar -----------

colors = ['#f0a8a8', '#FFBF00']  # con costa, sin costa

ax = df_stacked.plot(
    kind='barh',
    stacked=True,
    figsize=(10, 6),
    color=colors
)

ax.set_xlabel('Viajeros totales')
ax.set_ylabel('Año')
ax.set_title('FRO: Viajeros totales por año — CCAA con costa vs sin costa')
ax.legend(loc='lower right')
ax.grid(False)
plt.tight_layout()
plt.savefig('./img/viz-bar-anio-costa-no-costa.png')
plt.show()


# ##### VIZ CCAA: FRONTUR + ETR

# In[889]:


# calculamos el total para ello hacemos merge de los dos DS FRONTUR y ETR
# FRONTUR: resetear índice y derretir

df_frontur = pd.read_csv('./data/13884_15_25_ccaa.csv', index_col=0)
df_etr = pd.read_csv('./data/df_filtrado_14297_pivot_dest.csv', index_col=0)

frontur = df_frontur.reset_index().melt(id_vars=['Periodo'], 
                                         var_name='Comunidad autónoma', 
                                         value_name='Frontur')

# ETR: resetear índice y derretir
etr = df_etr.reset_index().melt(id_vars=['Periodo'], 
                                var_name='Comunidad autónoma', 
                                value_name='ETR')


# In[891]:


# Fusionar por Periodo y Comunidad Autónoma
h6_df_merged = pd.merge(frontur, etr, on=['Periodo', 'Comunidad autónoma'], how='outer')
h6_df_merged


# In[902]:


h6_df_merged['Comunidad autónoma'].unique()


# In[1072]:


import pandas as pd
import matplotlib.pyplot as plt

# Si no lo has hecho, carga o define tu DataFrame:
# df_total_h6 = pd.read_csv('ruta_al_archivo.csv')

# Limpia el nombre de la comunidad y ponlo en minúsculas
h6_df_merged['Comunidad autónoma'] = h6_df_merged['Comunidad autónoma'].str.replace(r'^\d{2} ', '', regex=True).str.strip().str.lower()

# Suma total por comunidad autónoma (ETR + FRONTUR)
h6_df_merged['Total'] = h6_df_merged[['Frontur', 'ETR']].sum(axis=1, skipna=True)
viajeros_por_ccaa = h6_df_merged.groupby('Comunidad autónoma')['Total'].sum()

ccaa_con_costa = [ccaa.lower() for ccaa in ccaa_con_costa]

# Total por grupo
total_con_costa = viajeros_por_ccaa[viajeros_por_ccaa.index.isin(ccaa_con_costa)].sum()
total_sin_costa = viajeros_por_ccaa[~viajeros_por_ccaa.index.isin(ccaa_con_costa)].sum()

# Serie resumen para el gráfico
serie_donut = pd.Series({'Con costa': total_con_costa, 'Sin costa': total_sin_costa})

# Gráfico donut
colors = ["#D7FBE1", "#D9A0E8"]

fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(
    serie_donut,
    labels=serie_donut.index,
    # autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*serie_donut.sum()):,})",
    autopct='%.1f%%',
    pctdistance=0.75,
    startangle=90,
    colors=colors,
    wedgeprops=dict(width=0.4)
)

# Ajustes estéticos
for text in texts + autotexts:
    text.set_fontsize(12)

ax.set_title('Distribución total de viajeros (ETR + FRONTUR): CCAA con costa vs sin costa')
plt.tight_layout()
plt.savefig('./img/viz-distr-viajeros_etr_fro_costa.png')
plt.show()


# In[908]:


viajeros_por_ccaa


# ##### VIZ Mapa Viajeros de costa

# In[943]:


import folium
import json
from unidecode import unidecode
import pandas as pd

# Cargar y preparar datos
df_mapa = viajeros_por_ccaa.reset_index()
df_mapa = df_mapa[~df_mapa['Comunidad autónoma'].isin(['anio', 'mes', 'españa', 'extranjera'])]
df_mapa.columns = ['Comunidad Autónoma', 'Total']
df_mapa['Comunidad Autónoma'] = df_mapa['Comunidad Autónoma'].str.strip().str.lower().apply(unidecode)

# Corrección de nombres para que coincidan con el GeoJSON
correccion_nombres = {
    'balears, illes': 'illes balears',
    'madrid, comunidad de': 'comunidad de madrid',
    'murcia, region de': 'region de murcia',
    'navarra, comunidad foral de': 'comunidad foral de navarra'
}
df_mapa['Comunidad Autónoma'] = df_mapa['Comunidad Autónoma'].replace(correccion_nombres)

# Normalizar columnas para que coincidan con el GeoJSON
with open('./data/provincias-espanolas 2.0.geojson') as f:
    geo = json.load(f)

# Normalizar valores dentro del geojson
for feature in geo['features']:
    feature['properties']['ccaa'] = unidecode(feature['properties']['ccaa'].lower().strip())
    ccaa = unidecode(feature['properties']['ccaa'].lower().strip())
    feature['properties']['ccaa'] = ccaa

    # Añadir valor de viajeros al GeoJSON para tooltip
    valor = df_mapa.loc[df_mapa['Comunidad Autónoma'] == ccaa, 'Total']
    feature['properties']['viajeros'] = f"{int(valor.values[0]):,}" if not valor.empty else "Sin datos"

# Crear mapa base
m = folium.Map(location=[40.4380986, -3.844343], zoom_start=5.5)

# Añadir capas base
tiles = ['cartodbpositron', 'cartodbdark_matter', 'openstreetmap']
for tile in tiles:
    folium.TileLayer(tile).add_to(m)

# Crear la coropleta
folium.Choropleth(
    geo_data=geo,
    name='choropleth',
    data=df_mapa,
    columns=['Comunidad Autónoma', 'Total'],
    key_on='feature.properties.ccaa',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Viajeros por comunidad autónoma'
).add_to(m)

# Tooltips personalizados
folium.GeoJson(
    geo,
    name='labels',
    style_function=lambda x: {"fillOpacity": 0, "color": "transparent", "weight": 0},
    tooltip=folium.GeoJsonTooltip(
        fields=["ccaa", "viajeros"],
        aliases=["Comunidad Autónoma:", "Total de viajeros:"],
        localize=True
    )
).add_to(m)

folium.LayerControl().add_to(m)
m


# ## H7 Ciertas comunidades autónomas han experimentado un mayor crecimiento turístico.

# ##### VIZ Facet grid

# In[951]:


h6_ccaa_growth = h6_df_merged.copy()


# In[952]:


h6_df_merged


# In[1073]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 1. Preprocesamiento
# -----------------------

# Convertir 'Periodo' a fecha
h6_ccaa_growth['fecha'] = pd.to_datetime(h6_ccaa_growth['Periodo'], format='%YM%m')
h6_ccaa_growth['anio'] = h6_ccaa_growth['fecha'].dt.year
h6_ccaa_growth['mes'] = h6_ccaa_growth['fecha'].dt.month

# Eliminar filas que no son CCAA (como 'españa', 'extranjera', etc.)
ccaa_validas = [
    'andalucía', 'aragón', 'asturias, principado de', 'balears, illes', 'canarias', 'cantabria',
    'castilla - la mancha', 'castilla y leon', 'cataluña', 'comunitat valenciana',
    'extremadura', 'galicia', 'madrid, comunidad de', 'murcia, región de',
    'navarra, comunidad foral de', 'país vasco', 'rioja, la'
]
h6_ccaa_growth = h6_ccaa_growth[h6_ccaa_growth['Comunidad autónoma'].str.lower().isin(ccaa_validas)]


# Creamos la columna 'mes_anio' para el eje X (formato "2020-03")
h6_ccaa_growth['mes_anio'] = h6_ccaa_growth['fecha'].dt.to_period('M').astype(str)

# Normalizamos valores por CCAA: escalar cada serie entre 0 y 1
h6_ccaa_growth['Total_norm'] = (
    h6_ccaa_growth
    .groupby('Comunidad autónoma')['Total']
    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
)

# Convertir 'mes_anio' a categoría ordenada
h6_ccaa_growth['mes_anio'] = pd.Categorical(
    h6_ccaa_growth['mes_anio'],
    categories=sorted(h6_ccaa_growth['mes_anio'].unique()),
    ordered=True
)

g = sns.FacetGrid(
    h6_ccaa_growth,
    col='Comunidad autónoma',
    col_wrap=2,
    sharey=False,
    height=3,
    aspect=1.8
)

g.map(sns.lineplot, 'mes_anio', 'Total_norm')

# Obtener los valores únicos ordenados del eje X
x_labels = h6_ccaa_growth['mes_anio'].cat.categories

for ax in g.axes.flatten():
    try:
        # # Extraer los valores x reales usados en la gráfica
        # line = ax.get_lines()[0]
        # x_vals = line.get_xdata()

        # Identificar las posiciones de abril y junio 2020
        start_idx = list(x_labels).index('2020-04')
        end_idx = list(x_labels).index('2020-06')

        # Marcar la zona en todos los subgráficos
        ax.axvspan(start_idx, end_idx, color='red', alpha=0.1)
        ax.grid(False)
    except Exception as e:
        print(f"Error en gráfico: {e}")
        continue


g.set_titles("{col_name}")
g.set_axis_labels("Mes-Año", "Viajes normalizados")
g.fig.suptitle("Evolución mensual de viajes por CCAA (normalizado)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./img/viz-evol-mens-_viajes-ccaa.png')
plt.show()


# In[ ]:




