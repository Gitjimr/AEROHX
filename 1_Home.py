import streamlit as st
st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v

from PIL import Image
import os
path = os.path.dirname(__file__)

st.set_page_config(
    page_title='AEROHX',
    layout="wide"
                   )

hide_menu = '''
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        '''
st.markdown(hide_menu, unsafe_allow_html=True)


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cadquery as cq
from cadquery import exporters

import pyvista as pv
import plotly.graph_objects as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os  # Required to check if the file exists
from os import listdir
from os.path import isfile, join

import cv2  #OpenCV

import streamlit_stl
from streamlit_stl import stl_from_file

import io

################# FUNCTIONS #################

def scale_xy_airfoil(points, h_c, chord):
  # Separar coordenadas x e y
  #x_coords, y_coords = zip(*points)

  # Criar o gráfico
  #plt.figure(figsize=(10, 5))
  #plt.plot(x_coords, y_coords, marker='o', markersize=2, linewidth=1)
  #plt.title("Curva gerada pelos pontos fornecidos")
  #plt.xlabel("x")
  #plt.ylabel("y")
  #plt.grid(True)
  #plt.axis("equal")
  #plt.show()

  height = h_c*chord

  max_y = max(points, key=lambda point: point[1])[1]
  min_y = min(points, key=lambda point: point[1])[1]
  max_height = max_y - min_y

  pontos = [(x * chord, y * (height/max_height)) for x, y in points]

  # Separar coordenadas x e y
  #x_coords, y_coords = zip(*pontos)

  # Criar o gráfico
  #plt.figure(figsize=(10, 5))
  #plt.plot(x_coords, y_coords, marker='o', markersize=2, linewidth=1)
  #plt.title("Curva gerada pelos pontos fornecidos")
  #plt.xlabel("x")
  #plt.ylabel("y")
  #plt.grid(True)
  #plt.axis("equal")
  #plt.show()

  return pontos


def calcular_dimensoes(c, h_c, p_v_h, p_h_c):
    espessura_h = h_c * c
    pitch_vertical = p_v_h * espessura_h
    pitch_horizontal = p_h_c * c
    print(f"Espessura: {espessura_h}, Pitch vertical: {pitch_vertical}, Pitch horizontal: {pitch_horizontal}")
    return espessura_h, pitch_vertical, pitch_horizontal


# Função para calcular coordenadas com deslocamentos alternados
def calcular_coordenadas_aerofolios(points, num_rows, num_cols, chord_length, h_c, p_v_h, p_h_c):
    # Calcular dimensões reais
    espessura_h, pitch_v, pitch_h = calcular_dimensoes(chord_length, h_c, p_v_h, p_h_c)

    # Separar as coordenadas dos pontos fornecidos
    x_coords, y_coords = zip(*points)

    all_contours = []

    # Calcular cada aerofólio com o deslocamento necessário
    for row in range(num_rows):
        for col in range(num_cols):
            # Deslocamentos horizontais e verticais
            x_offset = col * pitch_h
            y_offset = row * pitch_v * 2 + (pitch_v if col % 2 == 1 else 0)

            # Aplicar os deslocamentos aos pontos
            x_new = [x + x_offset for x in x_coords]
            y_new = [y + y_offset for y in y_coords]

            contour = [(x, y) for x, y in zip(x_new, y_new)]
            all_contours.append(contour)

    return all_contours, espessura_h, pitch_v, pitch_h


# Função apenas para plotar os aerofólios (mat plt lib)
def plot_airfoils_alternados(contours):
    plt.figure(figsize=(12, 6))

    # Plotar cada contorno fornecido
    for contour in contours:
        x_coords, y_coords = zip(*contour)
        plt.plot(x_coords, y_coords, marker='o', markersize=2, linewidth=1)

    plt.title("Distribuição Alternada de Aerofólios")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def process_coord_files(file, file_name):
    df, pontos = None, None

    if file_name.endswith('.xlsx'):
        try:
            excel_data = pd.read_excel(file, sheet_name=None)  # Load all sheets
            _sheet = list(excel_data.keys())[0]
            df = excel_data[_sheet]

            # Process columns
            if isinstance(list(df.columns)[0], str):
                df = df.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
            elif isinstance(list(df.columns)[0], (int, float)):
                df_part1 = pd.DataFrame({'x': [list(df.columns)[0]], 'y': [list(df.columns)[1]]})
                df_part2 = df.iloc[0:]
                df_part2 = df_part2.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
                df = pd.concat([pd.DataFrame(df_part1), df_part2], ignore_index=True)

            df = df.dropna().reset_index(drop=True)
            pontos = [(round(df['x'][i], 6), round(df['y'][i], 6)) for i in range(len(df['x']))]

        except Exception as e:
            st.error(f"Failed to read the Excel file: {str(e)}")

    elif file_name.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            column_data = list(df.columns)[0]

            try:
                column_data = float(column_data)
            except ValueError:
                pass

            # Process columns
            if isinstance(column_data, str):
                df = df.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
            elif isinstance(column_data, (int, float)):
                df_part1 = pd.DataFrame({'x': [list(df.columns)[0]], 'y': [list(df.columns)[1]]})
                df_part2 = df.iloc[0:]
                df_part2 = df_part2.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
                df = pd.concat([pd.DataFrame(df_part1), df_part2], ignore_index=True)

            df = df.dropna().reset_index(drop=True)
            pontos = [(round(float(df['x'][i]), 6), round(float(df['y'][i]), 6)) for i in range(len(df['x']))]

        except Exception as e:
            st.error(f"Failed to read the CSV file: {str(e)}")

    else:
        st.warning("Please upload a valid file in .xlsx or .csv format.")

    return df


@st.dialog("Uploading a Coordinates File")
def show_uploading_instructions():
    st.markdown("""
    - The file must be saved in either **.xlsx** or **.csv** format.  
    - For **.xlsx** files, column A represents the x-axis, while column B represents the y-axis.  
    - For **.xlsx** files, decimal values should use a `.` as the separator, not a `,`.  
    - For **.csv** files, the first value represents the x-axis, and the second value represents the y-axis.  
    - **Coordinates in both formats must be in order, with each (x, y) point located between the previous and next points.**
    """)
    st.image(path+'/pages/images/example_mhcad.png')


def scale_contour_df(df, scale=1):
    df = df.copy().astype(float)
    #center = df[['x', 'y']].mean(axis=0)
    #scaled_df = (df[['x', 'y']] - center) * scale + center
    scaled_df = (df[['x', 'y']]) * scale
    df[['x', 'y']] = scaled_df
    return df


@st.dialog("Sketch Preview ")
def plot_airfoils_alternados_plotly(contours):
    # Lista para armazenar os gráficos dos contornos
    fig = make_subplots(rows=1, cols=1)

    for i, contour in enumerate(contours):
        x_coords, y_coords = zip(*contour)

        # Adicionar uma linha para cada contorno
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            marker=dict(size=3),
            name=f'Aerofólio {i + 1}'
        ))

    # Ajustar layout
    fig.update_layout(
        title='Distribuição Alternada de Aerofólios',
        xaxis_title='x (mm)',
        yaxis_title='y (mm)',
        showlegend=False,
        width=800,
        height=600,
        template='plotly_white'#
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # Manter escala igual nos eixos
    #fig.show()
    st.plotly_chart(fig,  use_container_width=True)


def manter_extrude_button_ativo():
    st.session_state.extrude_button = True

def desativ_extrude_button():
    st.session_state.extrude_button = False


############################################################


#st.title("AEROHX", anchor=False)

col1, col2 = st.columns([1, 2])

#SETUP

if 'active_page' not in st.session_state:
    st.session_state.active_page = '1_Home'

    st.session_state.st_coordinates_x = []
    st.session_state.st_coordinates_y = []
    st.session_state.st_coordinates = []
    st.session_state.st_upload_coordinates = False

    st.session_state.st_chord = 1.
    st.session_state.st_h_c = 0.05
    st.session_state.st_p_v_h = 1.8
    st.session_state.st_p_h_c = 0.6
    st.session_state.st_num_rows = 6
    st.session_state.st_num_cols = 4
    st.session_state.st_length = 3.

    st.session_state.st_set_scale = False
    st.session_state.st_scale = 2.

    st.session_state.extrude_button = False

    st.session_state.st_sketch = None
    st.session_state.st_solid = None


col1.subheader('Extrusão de Contorno'
               ,anchor=False)

upload_coordinates = col1.toggle("Upload de Arquivo com Coordenadas", help='Para algum outro tipo de contorno específico para as tubulações.'
                                          ' Os pontos precisam estar ordenados.', key='st_upload_coordinates',on_change =desativ_extrude_button)

if not upload_coordinates:
    df__coord_input = process_coord_files(path+'/coord_originais_.xlsx', 'coord_originais_.xlsx')
    #col1.dataframe(df__coord_input)

else:
    if col1.button('Instruções',use_container_width=True):
        show_uploading_instructions()

    uploaded_file = col1.file_uploader('Arquivo em **.xlsx** ou **.csv**.',type=["xlsx","csv"])
    if uploaded_file is not None:
        with st.spinner('Carregando...'):
            df__coord_input = process_coord_files(uploaded_file, str(uploaded_file.name))
            col1.dataframe(df__coord_input)


## Entradas - Aerofólios

chord = col1.number_input("Tamanho de corda",format='%f',step=1.,min_value=1.,max_value=15.,key='st_chord',help='Tamanho da corda (mm)')

h_c = col1.number_input("Espessura adimensional",format='%f',step=0.01,min_value=0.05,max_value=0.25,key='st_h_c')

p_v_h = col1.number_input("Pitch vertical adimensional",format='%f',step=0.1,min_value=1.8,max_value=10.,key='st_p_v_h')

p_h_c = col1.number_input("Pitch horizontal adimensional",format='%f',step=0.1,min_value=0.6,max_value=5.,key='st_p_h_c')

num_rows = col1.number_input("Número de fileiras de aerofólios",step=1,min_value=1,key='st_num_rows',help='Número de fileiras de aerofólios')

num_cols = col1.number_input("Número de colunas de aerofólios",step=1,min_value=1,key='st_num_cols',help='Número de colunas de aerofólios')

length = col1.number_input("Comprimento de tubos",format='%f',step=1.,min_value=0.1,key='st_length',help='Comprimento de tubulações (mm)')

##


## Offset

offset_ = col1.toggle("Opcional: Criar uma espessura no modelo.", help='Criar volume oco -> Ativar esta opção.'
                                          , value=True)

##


## Set Scale Input

set_scale = col1.toggle("Opcional: Escalar", help='Essa opção permite escalar o modelo gerado proporcionalmente em x,y,z.'
                                          , value=False)

if not set_scale:
    set_scale = st.session_state['st_set_scale']
    scale = st.session_state['st_scale']
else:
    scale = col1.number_input("Escala",format='%f',step=0.5,min_value=0.01,key='st_scale')

##


## Sketch Preview

preview_button = col1.button("Sketch Preview",use_container_width = True, on_click=desativ_extrude_button)

if preview_button:
    try:
        scale_ = 1 if not set_scale else scale
        coordinates = [(round(float(df__coord_input['x'][i]), 8), round(float(df__coord_input['y'][i]), 8)) for i in
                       range(len(df__coord_input['x']))]

        pontos = scale_xy_airfoil(coordinates, h_c, chord)

        contours, espessura_h, pitch_v, pitch_h = calcular_coordenadas_aerofolios(pontos, num_rows, num_cols, chord,
                                                                                  h_c, p_v_h, p_h_c)
        plot_airfoils_alternados_plotly(contours)
    except Exception as e:
        col1.error(f"Erro em gerar os contornos: {e}")

##


run_button = col2.button("Gerar Modelo",use_container_width = True)

if run_button:
    st.session_state.extrude_button = True

if st.session_state.extrude_button:
  # Saving the inputs
  ## For coord's df
  try:
      st.session_state['st_coordinates_x'] = df__coord_input['x']
      st.session_state['st_coordinates_y'] = df__coord_input['y']
  except:
      col2.error('Erro: Coordenadas do Contorno.')
      pass

  with col2:

    with st.spinner('Carregando...'):

        try:

            ################# RUNNING #################

            coordinates = [(round(float(df__coord_input['x'][i]), 8), round(float(df__coord_input['y'][i]), 8)) for i in range(len(df__coord_input['x']))]

            pontos = scale_xy_airfoil(coordinates, h_c, chord)

            contours, espessura_h, pitch_v, pitch_h = calcular_coordenadas_aerofolios(pontos, num_rows, num_cols, chord,
                                                                                      h_c, p_v_h, p_h_c)

            contour_coordinates_ = contours

            result = None
            modelo_combinado = None

            for contour in contour_coordinates_:

                if contour is None:
                    continue

                ## Criar o esboço inicial
                sketch1 = cq.Sketch()

                try:
                    for i in range(len(contour) - 1):
                        sketch1 = sketch1.segment(contour[i], contour[i + 1])

                    sketch1 = sketch1.close().assemble(tag="face").reset()

                    if offset_ is True:
                        sketch1_offset = sketch1.copy().wires().offset(-(espessura_h * 0.1), mode='r').reset()

                    try:
                        result = result + sketch1 - sketch1_offset if offset_ is True else result + sketch1
                    except:
                        result = sketch1 - sketch1_offset if offset_ is True else sketch1

                except:
                    col2.error("Erro ao gerar contorno do modelo.")
                    continue

            if set_scale is True:
                try:
                    result = result.val().scale(scale)
                except:
                    col2.error("Erro ao escalar modelo.")
                    length = length / scale
                    # pass

            ## Exportar o modelo como STL
            exporters.export(result, 'sketch_hxairfoils.stl')

            ## 3D

            try:
                sketch = cq.Workplane("XY").placeSketch(result)

                modelo_combinado = sketch.extrude(length * scale if set_scale is True else length)

            except:
                col2.error("Erro ao gerar modelo 3D.")

            ### Exportar como STL
            exporters.export(modelo_combinado, 'solid_hxairfoils.stl')

            ############################### RESULTS

            try:
                stl_from_file(
                    file_path='solid_hxairfoils.stl',
                    material='material',
                    auto_rotate=False,
                    opacity=1,
                    cam_h_angle=90,
                    height=610,
                    max_view_distance=100000,
                    color='#4169E1'
                )
                st.success("Modelo gerado com sucesso.")
                st.session_state.st_sketch = result
                st.session_state.st_solid = modelo_combinado

            except:
                pass

            ############################### DOWNLOAD

            st.divider()
            st.subheader("⬇️ Download", divider='gray', anchor=False)

            col21, col22 = col2.columns([1, 1])

            solid_name = 'solid_hxairfoils.stl'
            sketch_name = 'sketch_hxairfoils.stl'
            stl_file_sketch = str(path) + "/" + sketch_name
            stl_file_solid = str(path) + "/" + solid_name

            # Create a download button for STL
            ## Sketch
            col21.download_button(
                label="Sketch Surface .stl",
                data=open(stl_file_sketch, "rb").read(),
                file_name=sketch_name,
                mime="application/stl",
                on_click=manter_extrude_button_ativo,
                use_container_width=True
            )
            ## Solid
            col22.download_button(
                label="Solid .stl",
                data=open(stl_file_solid, "rb").read(),
                file_name=solid_name,
                mime="application/stl",
                on_click=manter_extrude_button_ativo,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Erro ao gerar modelo: {e}")
else:
  col2.markdown("")
