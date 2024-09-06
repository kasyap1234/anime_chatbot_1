import pandas as pd
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import Ollama
from llama_index.embeddings import OllamaEmbedding
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama as LangChainOllama



def load_anime_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# Create a vector store index for anime recommendations
def create_anime_index(df):
    llm = Ollama(model="llama2")
    embed_model = OllamaEmbedding(model_name="llama2")
    documents = [doc for doc in df.to_dict('records')]
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)
    return index


# Anime recommendation function
def get_anime_recommendations(index, query, num_results=5):
    query_engine = index.as_query_engine()
    response = query_engine.query(f"Recommend {num_results} anime similar to: {query}")
    return response.response


# Story generator function using LangChain
def generate_anime_story(prompt, llm):
    story_template = """
    Create a short anime story based on the following prompt:
    {prompt}

    The story should include:
    1. A brief introduction of the main character(s)
    2. The setting of the story
    3. A conflict or challenge
    4. A resolution or cliffhanger ending

    Story:
    """
    story_prompt = PromptTemplate(template=story_template, input_variables=["prompt"])
    story_chain = LLMChain(llm=llm, prompt=story_prompt)
    return story_chain.run(prompt)


# Main application
def main():
    # Load anime data
    anime_df = load_anime_data("anime_data.csv")

    # Create anime index
    anime_index = create_anime_index(anime_df)

    # Initialize Ollama LLM for story generation
    llm = LangChainOllama(model="llama2")

    while True:
        print("\nAnime Recommendation and Story Generator")
        print("1. Get anime recommendations")
        print("2. Generate anime story")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            query = input("Enter an anime title or description for recommendations: ")
            recommendations = get_anime_recommendations(anime_index, query)
            print("\nRecommended Anime:")
            print(recommendations)

        elif choice == "2":
            prompt = input("Enter a prompt for the anime story: ")
            story = generate_anime_story(prompt, llm)
            print("\nGenerated Anime Story:")
            print(story)

        elif choice == "3":
            print("Thank you for using the Anime Recommendation and Story Generator. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
