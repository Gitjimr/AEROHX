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
import time

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


def centralizar_pontos_na_origem(pontos):
    if not pontos:
        return []

    # Separando as coordenadas X e Y
    x_coords, y_coords = zip(*pontos)

    # Calculando os valores máximos e mínimos
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Calculando os centros baseados nas distâncias máximas
    x_centro = (x_max + x_min) / 2
    y_centro = (y_max + y_min) / 2

    # Centralizando os pontos subtraindo o centro calculado
    pontos_centralizados = [(x - x_centro, y - y_centro) for x, y in pontos]

    return pontos_centralizados


def detectar_simetria(pontos):
    pontos_set = set(pontos)

    simetria_x = True
    simetria_y = True

    for x, y in pontos:
        if (x, -y) not in pontos_set:
            simetria_x = False
        if (-x, y) not in pontos_set:
            simetria_y = False

    return {
        'x': simetria_x,
        'y': simetria_y
    }


def raio_eq_poligono(pontos): # Para modificar perfil assimétrico nas curvas
    n = len(pontos)
    area = 0
    for i in range(n):
        x1, y1 = pontos[i]
        x2, y2 = pontos[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)

    area = abs(area) / 2
    raio_circ = (area / math.pi) ** 0.5

    return raio_circ


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

def suavizar_contorno(coordinates, suavizar, suavizacao):
    # Suavização **
    if suavizar is True:
        # Converter a lista de pontos para um array NumPy com o formato adequado para contornos do OpenCV
        contour = np.array(coordinates, dtype=np.float32).reshape((-1, 1, 2))

        # Calcular o perímetro do contorno
        peri = cv2.arcLength(contour, True)

        # Suavizar o contorno usando approxPolyDP
        epsilon = suavizacao * peri  # ajuste esse valor conforme necessário
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Converter o contorno suavizado para uma lista de tuplas (x, y)
        smoothed_points = [(float(x), float(y)) for [x, y] in smoothed_contour[:, 0]]

        coordinates = smoothed_points

    return coordinates

def hx_lin(
    modelo_combinado,
    s1,
    pitch_h_,
    pitch_v_,
    length_,
    num_cols,
    num_rows,
    set_scale,
    simetry,
    staggered = True
):
    if simetry['x']:
      #   CONEXÕES ENTRE LINHAS (PERFIL SIMÉTRICO)

      result_cima = (
          cq.Workplane("XY")
          .placeSketch(s1)
          .revolve(
              angleDegrees=180,
              axisStart=(1, pitch_v_, 0),  # mesmo X e Z
              axisEnd=(0, pitch_v_, 0)  # variação apenas em Y
          )
      )

      #exporters.export(result_cima, "revolve_cima.stl")

      result_baixo = (
          cq.Workplane("XY")
          .placeSketch(s1)
          .revolve(
              angleDegrees=180,
              axisStart=(0, pitch_v_, 0),  # mesmo X e Z
              axisEnd=(1, pitch_v_, 0)  # variação apenas em Y
          )
      )

    else:
      # Fittings
      r_circ_loft = raio_eq_poligono(pontos) * scale if set_scale else raio_eq_poligono(pontos)

      L_curva_col = 3 * r_circ_loft

      sketch_tubo_curva_col = (
          cq.Sketch()
          .circle(r_circ_loft)
          .circle(r_circ_loft - espessura_offset, mode="s")
          .reset()
      )

      sketch_tubo_interno_curva_col = (
          cq.Sketch()
          .circle(r_circ_loft - espessura_offset)
          .reset()
      )

      ## Fitting cima
      curva_col = cq.Workplane("XY").placeSketch(s1, sketch_tubo_curva_col.moved(z=L_curva_col)).loft(combine=True)

      if offset_:
          curva_col_int = cq.Workplane("XY").placeSketch(s1_offset_list, sketch_tubo_interno_curva_col.moved(z=L_curva_col)).loft(combine=True)
          curva_col_cima = curva_col - curva_col_int
      else:
          curva_col_cima = curva_col

      ## Fitting baixo
      curva_col = cq.Workplane("XY").placeSketch(s1, sketch_tubo_curva_col.moved(z=-L_curva_col)).loft(combine=True)

      if offset_:
          curva_col_int = cq.Workplane("XY").placeSketch(s1_offset_list, sketch_tubo_interno_curva_col.moved(z=-L_curva_col)).loft(combine=True)
          curva_col_baixo = curva_col - curva_col_int
      else:
          curva_col_baixo = curva_col

      ## Curva cima
      result_cima = (
          cq.Workplane("XY")
          .placeSketch(sketch_tubo_curva_col)
          .revolve(
              angleDegrees=180,
              axisStart=(1, pitch_v_, 0),  # mesmo X e Z
              axisEnd=(0, pitch_v_, 0)     # variação apenas em Y
          )
          .translate((0, 0, L_curva_col))
      )

      result_cima = result_cima + curva_col_cima + curva_col_cima.translate((0,pitch_v_*2,0))


      ## Curva baixo
      result_baixo = (
          cq.Workplane("XY")
          .placeSketch(sketch_tubo_curva_col)
          .revolve(
              angleDegrees=180,
              axisStart=(0, pitch_v_, 0),  # mesmo X e Z
              axisEnd=(1, pitch_v_, 0)     # variação apenas em Y
          )
          .translate((0, 0, -L_curva_col))
      )

      result_baixo = result_baixo + curva_col_baixo + curva_col_baixo.translate((0,pitch_v_*2,0))


    #exporters.export(result_baixo, "revolve_baixo.stl")

    modelo_final = modelo_combinado

    for i in range(num_cols):

        for j in range(num_rows - 1): # Aqui muda de staggered para alligned

            if j % 2 == 0 and i % 2 == 0:
                add_curva = result_baixo.translate((pitch_h_ * i, pitch_v_ * 2 * j, 0))
            elif j % 2 != 0 and i % 2 == 0:
                add_curva = result_cima.translate(
                    (pitch_h_ * i, pitch_v_ * 2 * j, length_ if set_scale is True else length))

            elif j % 2 == 0 and i % 2 != 0:
                add_curva = result_baixo.translate((pitch_h_ * i, pitch_v_ * 2 * j + pitch_v_, 0))
            elif j % 2 != 0 and i % 2 != 0:
                add_curva = result_cima.translate((pitch_h_ * i, pitch_v_ * 2 * j + pitch_v_,
                                                   length_ if set_scale is True else length))

            modelo_final = modelo_final + add_curva

    return modelo_final


def hx_col(
    pontos,
    scale,
    set_scale,
    espessura_offset,
    s1,
    s1_offset_list,
    offset_,
    pitch_h_,
    pitch_v_,
    length_,
    num_cols,
    num_rows,
    modelo_final,
    simetry,
    staggered = True
):
    if simetry['y'] and not staggered:

        # Simétrico

        # Curvas

        theta_rad_curva = math.atan(pitch_v_ / pitch_h_)    # Para alternados

        dist_curva = pitch_v_ / math.sin(theta_rad_curva) if staggered else pitch_h_

        ## Curva cima
        curva_col_ = (
            cq.Workplane("XY")
            .placeSketch(s1)
            .revolve(
                angleDegrees=180,
                axisStart=(dist_curva / 2, 0, 0),  # mesmo X e Z
                axisEnd=(dist_curva / 2, 1, 0)     # variação apenas em Y
            )
        )
        curva_col_ = curva_col_.rotate((0, 0, 0), (0, 0, 1), math.degrees(-theta_rad_curva)) if staggered else curva_col_

        curva_col_cima = curva_col_

        ## Curva baixo
        curva_col_ = (
            cq.Workplane("XY")
            .placeSketch(s1)
            .revolve(
                angleDegrees=180,
                axisStart=(dist_curva / 2, 1, 0),  # mesmo X e Z
                axisEnd=(dist_curva / 2, 0, 0)     # variação apenas em Y
            )
        )
        curva_col_ = curva_col_.rotate((0, 0, 0), (0, 0, 1), math.degrees(theta_rad_curva)) if staggered else curva_col_

        curva_col_baixo = curva_col_


    else:

        # Assimétrico

        # Fittings
        r_circ_loft = raio_eq_poligono(pontos) * scale if set_scale else raio_eq_poligono(pontos)

        L_curva_col = 3 * r_circ_loft

        sketch_tubo_curva_col = (
            cq.Sketch()
            .circle(r_circ_loft)
            .circle(r_circ_loft - espessura_offset, mode="s")
            .reset()
        )

        sketch_tubo_interno_curva_col = (
            cq.Sketch()
            .circle(r_circ_loft - espessura_offset)
            .reset()
        )

        ## Fitting cima
        curva_col = cq.Workplane("XY").placeSketch(s1, sketch_tubo_curva_col.moved(z=L_curva_col)).loft(combine=True)

        if offset_:
            curva_col_int = cq.Workplane("XY").placeSketch(s1_offset_list, sketch_tubo_interno_curva_col.moved(z=L_curva_col)).loft(combine=True)
            curva_col_cima = curva_col - curva_col_int
        else:
            curva_col_cima = curva_col

        ## Fitting baixo
        curva_col = cq.Workplane("XY").placeSketch(s1, sketch_tubo_curva_col.moved(z=-L_curva_col)).loft(combine=True)

        if offset_:
            curva_col_int = cq.Workplane("XY").placeSketch(s1_offset_list, sketch_tubo_interno_curva_col.moved(z=-L_curva_col)).loft(combine=True)
            curva_col_baixo = curva_col - curva_col_int
        else:
            curva_col_baixo = curva_col

        # Curvas

        theta_rad_curva = math.atan(pitch_v_ / pitch_h_)    # Para alternados

        dist_curva = pitch_v_ / math.sin(theta_rad_curva) if staggered else pitch_h_

        ## Curva cima
        curva_col_ = (
            cq.Workplane("XY")
            .placeSketch(sketch_tubo_curva_col)
            .revolve(
                angleDegrees=180,
                axisStart=(dist_curva / 2, 0, 0),  # mesmo X e Z
                axisEnd=(dist_curva / 2, 1, 0)     # variação apenas em Y
            )
        )
        curva_col_ = curva_col_.rotate((0, 0, 0), (0, 0, 1), math.degrees(-theta_rad_curva)).translate((0, 0, L_curva_col)) if staggered else curva_col_.translate((0, 0, L_curva_col))

        curva_col_cima = curva_col_cima + curva_col_ + curva_col_cima.translate((pitch_h_, -pitch_v_, 0))

        ## Curva baixo
        curva_col_ = (
            cq.Workplane("XY")
            .placeSketch(sketch_tubo_curva_col)
            .revolve(
                angleDegrees=180,
                axisStart=(dist_curva / 2, 1, 0),  # mesmo X e Z
                axisEnd=(dist_curva / 2, 0, 0)     # variação apenas em Y
            )
        )
        curva_col_ = curva_col_.rotate((0, 0, 0), (0, 0, 1), math.degrees(theta_rad_curva)).translate((0, 0, -L_curva_col)) if staggered else curva_col_.translate((0, 0, -L_curva_col))

        curva_col_baixo = curva_col_baixo + curva_col_ + curva_col_baixo.translate((pitch_h_, pitch_v_, 0))

        #exporters.export(curva_col_cima, "curva_col_cima.stl")
        #displayCAD("/content/curva_col_cima.stl", "Sketch: sketch_TESTE.stl")

        #exporters.export(curva_col_baixo, "curva_col_baixo.stl")
        #displayCAD("/content/curva_col_baixo.stl", "Sketch: sketch_TESTE.stl")


    modelo_final2 = modelo_final

    for i in range(num_cols - 1):
        if i % 2 == 0:
            ## Translação para o ramo com rotação
            translacao_pipeline1 = cq.Vector(pitch_h_ * i, (pitch_v_ * (num_rows - 1)) * 2, 0)

            ## Use o objeto já rotacionado e apenas aplique a translação diferente
            pipeline1 = curva_col_baixo.translate(translacao_pipeline1)

            modelo_final2 = modelo_final2 + pipeline1

        else:
            ## Translação para o ramo sem rotação
            translacao_pipeline2 = cq.Vector(pitch_h_ * i, pitch_v_, length_)

            pipeline2 = curva_col_cima.translate(translacao_pipeline2)

            modelo_final2 = modelo_final2 + pipeline2

    return modelo_final2


def criar_header_hx(
    connectype,
    pitch_h_,
    pitch_v_,
    num_cols,
    num_rows,
    espessura_offset,
    espessura_h_,
    chord_,
    offset_,
    sketch_ext_list,
    scale,
    set_scale,
    L_header,
    set_fitting,
    diam_interno_fitting,
    espessura_fitting,
    L_fitting,
    modelo_combinado,
    length_
):

    if connectype != 'Cabeçote':
        return None

    # Cálculo das dimensões da base inferior do cabeçote
    L_cab_b = pitch_h_ * num_cols * 1.1
    H_cab_b = pitch_v_ * num_rows * 2.5
    S_cab_b = espessura_offset * 1.5

    if offset_ is True:
        # Cria a base inferior do cabeçote com arredondamento nos cantos verticais
        cab_baixo = (
            cq.Workplane("XY")
            .box(L_cab_b, H_cab_b, S_cab_b)
            .edges("|Z").fillet(chord_/2)
        ).translate((pitch_h_ * (num_cols-1) / 2, pitch_v_ * num_rows - espessura_h_, S_cab_b / 2))

        # Subtrai os perfis (furos) do cabeçote
        for profile in sketch_ext_list:
            s1_header = profile
            s1_header = s1_header.val().scale(scale) if set_scale else s1_header

            header_hole = (
                cq.Workplane("XY")
                .placeSketch(s1_header)
                .consolidateWires()
                .extrude(S_cab_b)
            )

            cab_baixo = cab_baixo - header_hole
    else:
        # Cria a base inferior do cabeçote sem furos
        cab_baixo = (
            cq.Workplane("XY")
            .box(L_cab_b, H_cab_b, S_cab_b)
            .edges("|Z").fillet(chord_/2)
        ).translate((pitch_h_ * (num_cols-1) / 2, pitch_v_ * num_rows - espessura_h_, S_cab_b / 2))

    # Cria o corpo superior do cabeçote
    header_body = (
        cq.Workplane("XY")
        .box(L_cab_b, H_cab_b, L_header)
    ).translate((pitch_h_ * (num_cols-1) / 2, pitch_v_ * num_rows - espessura_h_, -L_header / 2))

    ## Se offset_ for True, aplica shell interno antes do fillet
    header_body = header_body.faces("+Z").shell(-espessura_offset * 2).edges("|Z").fillet(chord_/2) if offset_ else header_body.edges("|Z").fillet(chord_/2)

    ## Junta base inferior ao corpo do cabeçote
    header_body = header_body + cab_baixo

    ## Pega o centro da bounding box do cabeçote
    center_header = header_body.val().BoundingBox().center  # .BoundingBox() retorna a caixa delimitadora de um objeto, o menor paralelepípedo retangular alinhado aos eixos que envolve completamente o sólido

    # Cria corpo inferior do cabeçote (espelhado)
    header_body2 = (
        cq.Workplane("XY")
        .box(L_cab_b, H_cab_b, L_header)
    ).translate((pitch_h_ * (num_cols-1) / 2, pitch_v_ * num_rows - espessura_h_, L_header / 2))

    ## Aplica shell se necessário
    header_body2 = header_body2.faces("-Z").shell(-espessura_offset * 2).edges("|Z").fillet(chord_/2) if offset_ else header_body2.edges("|Z").fillet(chord_/2)

    ## Junta base inferior
    header_body2 = header_body2 + cab_baixo

    ## Pega o centro da bounding box do segundo cabeçote
    center_header2 = header_body2.val().BoundingBox().center

    if set_fitting is True:
        # Cálculo do diâmetro externo do tubo
        diam_externo_fitting = diam_interno_fitting + espessura_fitting * 2

        # Cria sketch do tubo com espessura (anular)
        sketch_tubo = (
            cq.Sketch()
            .circle(diam_externo_fitting/2)
            .circle(diam_interno_fitting/2, mode="s")
            .reset()
        )

        # Sketch interno para corte
        sketch_tubo_interno = (
            cq.Sketch()
            .circle(diam_interno_fitting/2)
            .reset()
        )

        wp = cq.Workplane("XY")

        # Cria tubos e rota para ficar perpendicular ao cabeçote
        conexao_tubo_externa = wp.placeSketch(sketch_tubo).extrude(L_fitting).rotate((0, 0, 0), (0, 1, 0), 90)
        cut_conexao_tubo_externa = wp.placeSketch(sketch_tubo_interno).extrude(L_fitting).rotate((0, 0, 0), (0, 1, 0), 90)

        # Adiciona tubo ao primeiro cabeçote e subtrai interno
        header_body = header_body + conexao_tubo_externa.translate((center_header.x + L_cab_b / 2 - espessura_offset * 2, center_header.y, center_header.z)) \
                                    - cut_conexao_tubo_externa.translate((center_header.x + L_cab_b / 2 - espessura_offset * 2, center_header.y, center_header.z))

        # Adiciona tubo ao segundo cabeçote e subtrai interno
        header_body2 = header_body2 + conexao_tubo_externa.translate((center_header2.x - L_cab_b / 2 + espessura_offset * 2 - L_fitting, center_header2.y, center_header2.z)) \
                                      - cut_conexao_tubo_externa.translate((center_header2.x - L_cab_b / 2 + espessura_offset * 2 - L_fitting, center_header2.y, center_header2.z))

    # Junta os dois cabeçotes e o modelo central
    modelo_final2 = header_body + modelo_combinado + header_body2.translate((0, 0, length_ - S_cab_b))

    # Exporta arquivos STL e STEP
    exporters.export(header_body, "header_body.step")

    return modelo_final2


def adicionar_fittings(
    modelo_final2,
    diam_interno_fitting,
    espessura_fitting,
    L_fitting,
    s1,
    s1_offset_list,
    set_fitting,
    num_cols,
    num_rows,
    pitch_h_,
    pitch_v_,
    length_
):  #   CONEXÕES DE ENTRADA E SAÍDA (FITTINGS)
    if not set_fitting:
        return modelo_final2

    diam_externo_fitting = diam_interno_fitting + espessura_fitting * 2

    sketch_tubo = (
        cq.Sketch()
        .circle(diam_externo_fitting/2)
        .circle(diam_interno_fitting/2, mode="s")
        .reset()
    )

    sketch_tubo_interno = (
        cq.Sketch()
        .circle(diam_interno_fitting/2)
        .reset()
    )

    if connectype == 'Curvas':
        wp = cq.Workplane("XY")

        conexao_tubo_externa = wp.placeSketch(s1, sketch_tubo.moved(z=L_fitting)).loft(combine=True)

        conexao_tubo = conexao_tubo_externa + wp.placeSketch(
            sketch_tubo.moved(z=L_fitting)).extrude(L_fitting)

        try:
            conexao_tubo_interna = wp.placeSketch(sketch_tubo_interno.moved(z=L_fitting), s1_offset_list).loft(
                combine=True)

            conexao_tubo = conexao_tubo - conexao_tubo_interna
        except:
            pass

        conexao_tubo_rot = conexao_tubo.rotate((0, 0, 0), (1, 0, 0), 180)

        modelo_final3 = modelo_final2

        for i in range(num_cols):
            if i == 0:
                translacao_pipeline1 = cq.Vector(pitch_h_ * i, 0, length_)
                conexao_tubo1 = conexao_tubo.translate(translacao_pipeline1)
                modelo_final3 = modelo_final3 + conexao_tubo1

            elif i == num_cols - 1 and i % 2 != 0:
                translacao_pipeline2 = cq.Vector(pitch_h_ * i, pitch_v_, length_)
                conexao_tubo2 = conexao_tubo.translate(translacao_pipeline2)
                modelo_final3 = modelo_final3 + conexao_tubo2

            elif i == num_cols - 1 and i % 2 == 0:
                translacao_pipeline2 = cq.Vector(pitch_h_ * i, (pitch_v_ * (num_rows - 1)) * 2, 0)
                conexao_tubo2 = conexao_tubo_rot.rotate((0, 0, 0), (1, 0, 0), 180).translate(translacao_pipeline2)
                modelo_final3 = modelo_final3 + conexao_tubo2

        return modelo_final3
    else:
        return modelo_final2


############################################################


#st.title("AEROHX", anchor=False)

col1, col2 = st.columns([1, 2])

#SETUP

if 'active_page_2' not in st.session_state:

    st.session_state.active_page_2 = '2_Tubos_com_Conexoes'
    st.session_state.st_suavizar = True
    st.session_state.st_suavizacao = 0.0005

    #if 'active_page' not in st.session_state:

    st.session_state.st_coordinates_x = []
    st.session_state.st_coordinates_y = []
    st.session_state.st_coordinates = []
    st.session_state.st_upload_coordinates = False

    st.session_state.st_chord = 15.
    st.session_state.st_h_c = 0.25
    st.session_state.st_p_v_h = 2.
    st.session_state.st_p_h_c = 1.5
    st.session_state.st_num_rows = 4
    st.session_state.st_num_cols = 3
    st.session_state.st_length = 250.
    st.session_state.st_espessura_offset = 1.
    st.session_state.st_L_header = 12.

    st.session_state.extrude_button = False

    st.session_state.st_sketch = None
    st.session_state.st_solid = None

    st.session_state.st_set_scale = False
    st.session_state.st_scale = 1.

    st.session_state.st_diam_interno_fitting = 6.35
    st.session_state.st_espessura_fitting = 1.
    st.session_state.st_L_fitting = 20.


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
with col1.expander("Criar espessura no modelo"):

    offset_ = st.toggle("Opcional: Criar uma espessura no modelo.", help='Ative para criar volume oco. Observação: as medidas selecionadas anteriormente valerão para o perfil externo.'
                                              , value=True)
    if not offset_:
        espessura_offset = st.session_state['st_espessura_offset']
    else:
        espessura_offset = st.number_input("Espessura do perfil",format='%f',step=0.5,min_value=0.01,key='st_espessura_offset', help='Espessura do perfil do modelo.')

##


## Set Scale Input
with col1.expander("Aplicar escala"):

    set_scale = st.toggle("Opcional: Escalar (não interfere na espessura)", help='Essa opção permite escalar o modelo gerado proporcionalmente em x,y,z.'
                                              , value=False)

    if not set_scale:
        set_scale = st.session_state['st_set_scale']
        scale = 1
    else:
        scale = st.number_input("Escala",format='%f',step=0.5,min_value=0.01,key='st_scale')

##

## Selecting HX Tube Connections
with col1.expander("Tipo de conexões"):
    connectype = st.selectbox(
        "Tipo de conexão entre tubos",
        ("N/A", "Curvas", "Cabeçote"), index=0, help='Selecione como os tubos devem ser conectados uns aos outros'
    )

    if connectype != 'Cabeçote':
        L_header = st.session_state['st_L_header']
    else:
        L_header = st.number_input("Comprimento do Cabeçote", format='%f', step=1., key='st_L_header', help='Deve ser maior que o diâmetro de fitting, se esse existir.')

##

## Set Smooth Input

with col1.expander("Suavização"):

    suavizar = st.toggle("Opcional: Suavizar", help='Essa opção permite suavizar ligeiramente a curva do perfil e acelerar a geração da geometria.'
                                              , value=True)

    if not suavizar:
        suavizar = st.session_state['st_suavizar']
        suavizacao = st.session_state['st_suavizacao']
    else:
        suavizacao = st.number_input("Suavização",format='%f',step=0.0005,max_value=0.001,key='st_suavizacao')

##

## Fittings Entrada e Saída

with col1.expander("Fitting para entrada e saída"):

    set_fitting = st.toggle("Opcional: Inserir Fitting", help='Essa opção permite inserir uma tubulação circular na entrada e na saída do trocador.'
                                              , value=False)

    if not set_fitting:
        diam_interno_fitting = st.session_state['st_diam_interno_fitting']
        espessura_fitting = st.session_state['st_espessura_fitting']
        L_fitting = st.session_state['st_L_fitting']
    else:
        diam_interno_fitting = st.number_input("Diâmetro Interno Fitting", format='%f', step=1., key='st_diam_interno_fitting')
        espessura_fitting = st.number_input("Espessura Fitting", format='%f', step=1., key='st_espessura_fitting')
        L_fitting = st.number_input("Comprimento Fitting", format='%f', step=1., key='st_L_fitting')

##

## Sketch Preview

preview_button = col1.button("Sketch Preview",use_container_width = True, on_click=desativ_extrude_button)

if preview_button:
    try:
        scale_ = 1 if not set_scale else scale
        coordinates = [(round(float(df__coord_input['x'][i]), 8), round(float(df__coord_input['y'][i]), 8)) for i in
                       range(len(df__coord_input['x']))]

        coordinates = centralizar_pontos_na_origem(coordinates)

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
            my_bar = st.progress(0, text='Tubulações')

            ################# RUNNING #################

            coordinates = [(round(float(df__coord_input['x'][i]), 8), round(float(df__coord_input['y'][i]), 8)) for i in range(len(df__coord_input['x']))]

            coordinates = centralizar_pontos_na_origem(coordinates) #*

            simetry = detectar_simetria(coordinates)  # +

            coordinates = suavizar_contorno(coordinates, suavizar, suavizacao)

            pontos = scale_xy_airfoil(coordinates, h_c, chord)

            contours, espessura_h, pitch_v, pitch_h = calcular_coordenadas_aerofolios(pontos, num_rows, num_cols, chord,
                                                                                      h_c, p_v_h, p_h_c)

            contour_coordinates_ = contours

            result = None
            modelo_combinado = None

            sketch_list = []
            sketch_offset_list = []  #internal
            sketch_ext_list  = []   #external

            for contour in contour_coordinates_:

                if contour is None:
                    continue

                ## Criar o esboço inicial
                sketch1 = cq.Sketch()

                try:
                    for i in range(len(contour) - 1):
                        sketch1 = sketch1.segment(contour[i], contour[i + 1])

                    sketch1 = sketch1.close().assemble(
                        tag="face").reset()  # * RESET Limpa o estado interno do esboço, mantendo apenas o resultado final (a face)

                    sketch_ext_list.append(sketch1)

                    if offset_ is True:
                        try:
                            sketch1_offset = sketch1.copy().wires().offset(-(espessura_offset / scale),
                                                                           mode='r').reset() if set_scale else sketch1.copy().wires().offset(
                                -(espessura_offset),
                                mode='r').reset()  # * * RESET Limpa o estado interno do esboço, mantendo apenas o resultado final (a face)
                        except:
                            sketch1_offset = sketch1.copy().wires().offset(-(espessura_h * 0.1),
                                                                           mode='r').reset()  # * * RESET Limpa o estado interno do esboço, mantendo apenas o resultado final (a face)
                            print(f'Espessura do perfil modificada para:{espessura_h * 0.1}')

                        sketch_offset_list.append(sketch1_offset)  # testing

                    sketch_list.append(sketch1 - sketch1_offset if offset_ is True else sketch1)  # testing

                    try:
                        result = result + sketch1
                        if offset_ is True:
                            result_ = result_ + sketch1_offset  # *
                    except:
                        result = sketch1
                        if offset_ is True:
                            result_ = sketch1_offset  # *

                except:
                    print("Error generating contour sketch")
                    continue

            if offset_ is True:
                result = result - result_

            if set_scale is True:
                try:
                    result = result.val().scale(scale)  # .translate(((-chord/2)*scale, 0, 0))
                except:
                    print("Error during scaling")
                    length = length / scale
                    # pass


            # 3D

            ##   SCALE

            ### Pegar os sketches de referência
            s1 = sketch_list[0]
            try:
                s1_offset_list = sketch_offset_list[0]
            except:
                pass

            ### Copiar os valores iniciais de pitch
            pitch_v_ = pitch_v
            pitch_h_ = pitch_h
            chord_ = chord
            espessura_h_ = espessura_h
            length_ = length

            if set_scale:
                try:
                    s1 = s1.val().scale(scale)
                    s1_offset_list = s1_offset_list.val().scale(scale) if offset_ else 0
                    pitch_v_ = pitch_v_ * scale
                    pitch_h_ = pitch_h_ * scale
                    chord_ = chord_ * scale
                    espessura_h_ = espessura_h_ * scale
                    length_ = length * scale
                except Exception as e:
                    st.error(f"Scaling error: {e}")

            try:
                ## Exportar o modelo como STL

                exporters.export(result, 'sketch_hxairfoils.stl')

                sketch = cq.Workplane("XY").placeSketch(result)

                modelo_combinado = sketch.extrude(length_)

                tube_profile = cq.Workplane("XY").placeSketch(s1).extrude(length_)
                exporters.export(tube_profile, 'hx_profile.stl')
                if connectype == 'N/A':
                    exporters.export(tube_profile, 'hx_profile.step')


            except:
                col2.error("Erro ao gerar modelo 3D.")


            ################# CONEXÕES #################

            ################# CURVAS

            if connectype == 'Curvas':
                my_bar.progress(50, text='Conexões entre linhas')

                #   CONEXÕES ENTRE LINHAS

                modelo_final = hx_lin(
                        modelo_combinado,
                        s1,
                        pitch_h_,
                        pitch_v_,
                        length_,
                        num_cols,
                        num_rows,
                        set_scale,
                        simetry
                    )


                #   CONEXÕES ENTRE COLUNAS

                my_bar.progress(70, text='Conexões entre colunas')

                modelo_final2 = hx_col(
                      pontos=pontos,
                      scale=scale,
                      set_scale=set_scale,
                      espessura_offset=espessura_offset,
                      s1=s1,
                      s1_offset_list=s1_offset_list,
                      offset_=offset_,
                      pitch_h_=pitch_h_,
                      pitch_v_=pitch_v_,
                      length_=length_,
                      num_cols=num_cols,
                      num_rows=num_rows,
                      modelo_final=modelo_final,
                      simetry=simetry
                  )


                #   CONEXÕES DE ENTRADA E SAÍDA (FITTINGS)

                if set_fitting:
                    my_bar.progress(85, text='Adicionando conexões de entrada e saída')

                modelo_final2 = adicionar_fittings(
                    modelo_final2,
                    diam_interno_fitting,
                    espessura_fitting,
                    L_fitting,
                    s1,
                    s1_offset_list,
                    set_fitting,
                    num_cols,
                    num_rows,
                    pitch_h_,
                    pitch_v_,
                    length_
                )


            ################# CABEÇOTE

            elif connectype == 'Cabeçote':
                my_bar.progress(50, text='Criando Headers')

                modelo_final2 = criar_header_hx(
                    connectype,
                    pitch_h_,
                    pitch_v_,
                    num_cols,
                    num_rows,
                    espessura_offset,
                    espessura_h_,
                    chord_,
                    offset_,
                    sketch_ext_list,
                    scale,
                    set_scale,
                    L_header,
                    set_fitting,
                    diam_interno_fitting,
                    espessura_fitting,
                    L_fitting,
                    modelo_combinado,
                    length_
                )

            ################# N/A

            else:
                modelo_final2 = modelo_combinado


            ############################### RESULTS

            exporters.export(modelo_final2, 'hx_final.stl')
            exporters.export(modelo_final2, 'hx_final.step')

            my_bar.progress(90, text='Display')

            try:
                stl_from_file(
                    file_path='hx_final.stl',
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
                st.session_state.st_solid = modelo_final2

            except:
                pass

            my_bar.progress(95, text='Arquivos .STL')


            ############################### DOWNLOAD

            st.divider()
            st.subheader("⬇️ Download", divider='gray', anchor=False)

            col21, col22 = col2.columns([1, 1])

            solid_name = 'hx_final'
            sketch_name = 'sketch_hxairfoils.stl'
            profile_name = 'hx_profile'
            stl_file_sketch = str(path) + "/" + sketch_name
            stl_file_solid = str(path) + "/" + solid_name + '.stl'
            step_file_solid = str(path) + "/" + solid_name + '.step'
            stl_file_profile = str(path) + "/" + profile_name + '.stl'
            step_file_profile = str(path) + "/" + profile_name + '.step'

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
                file_name=solid_name + '.stl',
                mime="application/stl",
                on_click=manter_extrude_button_ativo,
                use_container_width=True
            )

            col2.download_button(
                label="Solid .step",
                data=open(step_file_solid, "rb").read(),
                file_name=solid_name + '.step',
                mime="application/step",
                on_click=manter_extrude_button_ativo,
                use_container_width=True
            )

            if connectype == 'Cabeçote':
                step_file_header = str(path) + "/" + 'header_body' + '.step'
                col2.download_button(
                    label="Header (Solid) .step",
                    data=open(step_file_header, "rb").read(),
                    file_name='header_body' + '.step',
                    mime="application/step",
                    on_click=manter_extrude_button_ativo,
                    use_container_width=True
                )

            if connectype == 'N/A':
                col21.download_button(
                    label="Tube Profile (Solid) .stl",
                    data=open(stl_file_profile, "rb").read(),
                    file_name=profile_name+'.stl',
                    mime="application/stl",
                    on_click=manter_extrude_button_ativo,
                    use_container_width=True
                )
                col22.download_button(
                    label="Tube Profile (Solid) .step",
                    data=open(step_file_profile, "rb").read(),
                    file_name=profile_name+'.step',
                    mime="application/step",
                    on_click=manter_extrude_button_ativo,
                    use_container_width=True
                )
            else:
                col2.download_button(
                    label="Tube Profile (Solid) .stl",
                    data=open(stl_file_profile, "rb").read(),
                    file_name=profile_name+'.stl',
                    mime="application/stl",
                    on_click=manter_extrude_button_ativo,
                    use_container_width=True
                )

            my_bar.progress(100, text='Carregando')

            time.sleep(1)
            my_bar.empty()

        except Exception as e:
            st.error(f"Erro ao gerar modelo: {e}")
else:
  col2.markdown("")

  #https://aerohx-xbgevddyrwrida74b4hvnx.streamlit.app/
