import streamlit as st
import utils
import pandas as pd

dataset_path = "/Users/asutoshdalei/Desktop/Masters/Padhai/MSML610/CLIPEmbed/flickr8k/"
dataset = pd.read_csv(dataset_path + "captions.txt")


st.title("Visual Search Engine using CLIP Embeddings.")

tab1, tab2, tab3 = st.tabs(
    [
        "Text Search",
        "Image Search",
        "About & Analysis",
    ]
)

with tab1:
    input_promt_text = st.chat_input("Type something")
    if input_promt_text:
        text_embedding = utils.get_text_embedding(input_promt_text)
        distances, indices = utils.search_vec_db(text_embedding, vectorDB="image", k=3)

        st.subheader(f"Top 3 Hits for the query: {input_promt_text}")

        t1, t2, t3 = st.columns(3, vertical_alignment="top")
        with t1:
            st.write(f"Top 1 Hit Distance: {distances[0][0]:.3f}")
            # row = dataset.iloc[indices[0][0]]
            row = dataset["image"].unique()[indices[0][0]]
            st.image(dataset_path + f"images/{row}")

        with t2:
            st.write(f"Top 2 Hit Distance: {distances[0][1]:.3f}")
            # row = dataset.iloc[indices[0][1]]
            row = dataset["image"].unique()[indices[0][1]]
            st.image(dataset_path + f"images/{row}")

        with t3:
            st.write(f"Top 3 Hit Distance: {distances[0][2]:.3f}")
            # row = dataset.iloc[indices[0][2]]
            row = dataset["image"].unique()[indices[0][2]]
            st.image(dataset_path + f"images/{row}")

with tab2:
    input_promt_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if input_promt_image:
        st.image(input_promt_image)
        bytes_data = input_promt_image.getvalue()
        image_embedding = utils.get_image_embedding_from_bytes(bytes_data)
        distances, indices = utils.search_vec_db(image_embedding, vectorDB="text", k=3)
        row = dataset["caption"].unique()[indices[0][1]]
        st.write(row)

with tab3:
    st.write(
        """This project leverages OpenAI’s CLIP (Contrastive Language–Image Pretraining)
        model to power a fast and intuitive visual search experience. By encoding both
        images and text into a shared embedding space, our system allows users to search
        for images using natural language queries"""
    )
