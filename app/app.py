import streamlit as st
from your_model import generate_recipe  # import your real model logic

st.title("🍽️ Smart Recipe Generator")

ingredients = st.text_area("Enter your ingredients (comma-separated):", "tomato, garlic, onion")
time = st.slider("How much time do you have? (minutes)", 5, 120, 30)
diet = st.selectbox("Dietary Preference", ["Any", "Vegan", "Vegetarian", "Non-Vegetarian"])

if st.button("Generate Recipe"):
    recipe = generate_recipe(ingredients, time, diet)
    st.success("Here's your recipe!")
    st.write(f"**Title:** {recipe['title']}")
    st.write("**Ingredients:**")
    st.write(", ".join(recipe["ingredients"]))
    st.write("**Instructions:**")
    for i, step in enumerate(recipe["instructions"]):
        st.write(f"{i+1}. {step}")




