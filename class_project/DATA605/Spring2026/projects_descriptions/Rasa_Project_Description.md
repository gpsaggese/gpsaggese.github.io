# Rasa

## Description
- Rasa is an open-source framework for building conversational AI, specifically
  chatbots and virtual assistants.
- It provides a natural language understanding (NLU) component that allows the
  system to comprehend user input and extract intents and entities.
- The dialogue management component enables developers to design complex
  conversational flows and handle multi-turn dialogues effectively.
- Rasa supports integration with various messaging platforms, making it easy to
  deploy chatbots on websites, mobile apps, and social media.
- The framework is highly customizable, allowing for the use of pre-trained
  models or fine-tuning with domain-specific data to enhance performance.

## Project Objective
The goal of this project is to build a conversational AI chatbot that can assist
users in finding information about various public datasets. The chatbot will be
able to understand user queries, provide relevant dataset suggestions, and
facilitate user engagement through a natural dialogue.

## Dataset Suggestions
1. **Kaggle Datasets**
   - **Source**: Kaggle
   - **URL**: [Kaggle Datasets](https://www.kaggle.com/datasets)
   - **Data Contains**: A variety of datasets across multiple domains, including
     health, finance, sports, and more.
   - **Access Requirements**: Free account creation on Kaggle.

2. **Open Government Data**
   - **Source**: Data.gov
   - **URL**: [Data.gov](https://www.data.gov/)
   - **Data Contains**: Public datasets from various U.S. government agencies
     covering topics like climate, education, and public health.
   - **Access Requirements**: No authentication required; datasets are publicly
     accessible.

3. **Hugging Face Datasets**
   - **Source**: Hugging Face
   - **URL**: [Hugging Face Datasets](https://huggingface.co/datasets)
   - **Data Contains**: A variety of datasets for NLP tasks, including text
     classification and summarization.
   - **Access Requirements**: Free to access, no authentication needed.

4. **World Health Organization (WHO)**
   - **Source**: WHO
   - **URL**: [WHO Data](https://www.who.int/data/gho)
   - **Data Contains**: Health-related datasets, including global health
     statistics and disease prevalence.
   - **Access Requirements**: Publicly available without authentication.

## Tasks
- **Set Up Rasa Environment**: Install Rasa and set up a new project to build
  your chatbot.
- **Define Intents and Entities**: Create a list of user intents (e.g., "find
  dataset", "get dataset details") and define relevant entities (e.g., dataset
  names, topics).
- **Train NLU Model**: Use the defined intents and entities to train the NLU
  model, enabling the chatbot to understand user queries.
- **Design Dialogue Flows**: Create conversation stories that define how the
  chatbot should respond to various user inputs and guide them through dataset
  discovery.
- **Integrate Dataset Suggestions**: Implement logic to fetch and present
  dataset suggestions based on user queries, utilizing the selected datasets.
- **Test and Evaluate**: Conduct user testing to evaluate the chatbot's
  performance, gather feedback, and refine the NLU model and dialogue flows.

## Bonus Ideas
- **Add Personalization**: Implement user session management to tailor dataset
  suggestions based on previous interactions.
- **Integrate with External APIs**: Allow the chatbot to pull real-time data
  from external APIs related to datasets, such as recent updates or metadata.
- **Deploy the Chatbot**: Publish the chatbot on a web platform or messaging
  service to allow real users to interact with it.
- **Create a Dashboard**: Build a simple dashboard to visualize user
  interactions and popular dataset queries.

## Useful Resources
- [Rasa Documentation](https://rasa.com/docs/rasa/)
- [Rasa GitHub Repository](https://github.com/RasaHQ/rasa)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Data.gov](https://www.data.gov/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
