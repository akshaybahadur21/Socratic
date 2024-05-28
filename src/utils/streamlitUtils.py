import os
import random
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from openai import OpenAI

from src.HandDetector import HandDetector


def set_streamlit_header():
    st.set_page_config(page_title="Socratic 🦉", layout="wide", page_icon="🦉")
    image = Image.open('resources/socratic.png')
    col1, col2, col3 = st.columns([4, 2, 4])
    with col1:
        st.write(' ')
    with col2:
        st.image(image, width=300, caption='A simple Mathematical utility for Humans')
    with col3:
        st.write(' ')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    return image


def set_streamlit_footer():
    col1, col2, col3 = st.columns([5, 3, 4])
    with col1:
        st.write(' ')
    with col2:
        st.markdown("""
                    ###### Made with ❤️ and 🦙 by [Akshay Bahadur](https://akshaybahadur.com)
                    """)
    with col3:
        st.write(' ')


def generate_user_prompt():
    prompt = r"""
             #### User Prompt:
             - You are Socratic 🦉 - an LLM powered mathematician for humans.  
             - You are given the figure of a geometrical structure with related inputs. 
             - You have to find the area of the figure and mention the steps
            """
    return prompt


def response_generator(response, wait_time, my_bar=None, progress_text=""):
    if my_bar:
        for percent_complete in range(100):
            time.sleep(random.uniform(0, 0.1))
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()
    for word in response.split(" "):
        yield word + " "
        time.sleep(wait_time)


def set_basic_config():
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"
    detector = HandDetector()
    brush_thick = 15
    eraser_thick = 40
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    options = "--psm 8"
    counter_map = {
        "erase": 0,
        "write": 0,
        "go": 0
    }
    blkboard = np.zeros((720, 1280, 3), np.uint8)

    return openai_client, detector, brush_thick, eraser_thick, rectKernel, options, counter_map, blkboard
