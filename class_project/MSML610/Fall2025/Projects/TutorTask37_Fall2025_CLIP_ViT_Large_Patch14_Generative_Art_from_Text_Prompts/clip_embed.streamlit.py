import pandas as pd
import streamlit as st

import clip_embed_utils as utils

dataset_path = "clip_embed.dataset/"
dataset = pd.read_csv(dataset_path + "captions.txt")


st.title("Visual Search Engine using CLIP Embeddings.")

tab1, tab2, tab3 = st.tabs(["About & Analysis", "Text Search", "Image Search"])

with tab1:
    st.write(
        """This project leverages OpenAI’s CLIP (Contrastive Language–Image Pretraining)
        model to power a fast and intuitive visual search experience. By encoding both
        images and text into a shared embedding space, our system allows users to search
        for images using natural language queries.
        """
    )
    st.image("ClipArchitecture.png")
    st.divider()
    st.write("""It enables two core capabilities:""")
    st.write(
        "Text → Image Retrieval: Input a text query and retrieve the top 3 most relevant images from a dataset"
    )
    st.write(
        "Image → Text Retrieval (Image Captioning): Input an image and receive the most semantically relevant caption or text description based on similarity in the embedding space."
    )

    st.write("""Traditional search engines rely on tags or fixed metadata, which often fail to capture visual meaning. CLIP bridges this gap by embedding images and text into the same semantic space.
    By adding FAISS (Facebook AI Similarity Search) as the indexing layer, the system becomes scalable anf fast.""")

with tab2:
    k_ = st.segmented_control(
        "Select k number",
        options=[1, 2, 3, 4, 5],
        default=1,
        selection_mode="single",
        key="txt2img",
    )
    input_promt_text = st.chat_input("Type something")
    if input_promt_text:
        text_embedding = utils.get_text_embedding(input_promt_text)
        distances, indices = utils.search_vec_db(text_embedding, vectorDB="image", k=k_)

        st.subheader(f"Top {k_} Hits for the query: {input_promt_text}")

        hits_ = st.columns(k_, vertical_alignment="top")
        i_ = 0
        for ht_ in hits_:
            with ht_:
                st.write(f"Top {i_ + 1} Hit Distance: {distances[0][i_]:.3f}")
                row = dataset["image"].unique()[indices[0][i_]]
                st.image(dataset_path + f"images/{row}")
            i_ += 1


with tab3:
    input_promt_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    k_ = st.segmented_control(
        "Select k number",
        options=[1, 2, 3, 4, 5],
        default=1,
        selection_mode="single",
        key="img2txt",
    )
    if input_promt_image:
        st.image(input_promt_image)
        bytes_data = input_promt_image.getvalue()
        image_embedding = utils.get_image_embedding_from_bytes(bytes_data)
        distances, indices = utils.search_vec_db(image_embedding, vectorDB="text", k=k_)
        for i in range(k_):
            row = dataset["caption"].unique()[indices[0][i]]
            st.container(border=True).write(
                f"{row} \n | Hit Distance: {distances[0][i]}"
            )
