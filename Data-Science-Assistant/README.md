# Data Science Query Assistant

The Data Science Query Assistant is a Streamlit application designed to provide detailed explanations, generate quizzes on various data science topics, analyze CSV datasets, and facilitate conversations with information sourced from websites. Utilizing the power of AI models from Cohere and EdenAI, this assistant aims to enhance learning and data analysis experiences.

## Features

- **Topic Explanation**: Offers detailed explanations on data science topics.
- **Quiz Generation**: Creates quizzes for better learning engagement.
- **CSV Dataset Analysis**: Analyzes uploaded CSV files to answer queries.
- **Website Conversation**: Allows users to chat with AI that retrieves information from specified websites.

## Installation

To run the Data Science Query Assistant, you need Python installed on your system. Follow these steps to set up the project environment:

1. **Clone the Repository**

```
git clone https://github.com/nikbearbrown/AI4ED.git
cd Data-Science-Assistant
```


2. **Install Requirements**

```
pip install -r requirements.txt
```


3. **Environment Variables**

Create a `.env` file in the root directory and add your Cohere and EdenAI API keys:

```
COHERE_API_KEY=your_cohere_api_key_here
EDENAI_API_KEY=your_edenai_api_key_here
```


4. **Running the Application**

```
streamlit run app.py
```


## Usage

- **Query Assistant Tab**: Type your data science-related questions to get detailed explanations or quizzes.
- **Load Datasets Tab**: Upload a CSV file and ask questions regarding the dataset for analysis.
- **Chat with Websites Tab**: Enter a website URL to enable the AI to fetch and utilize information from the website for conversation.

## Dependencies

- Streamlit
- Cohere
- EdenAI
- spaCy
- Python-dotenv

## Contributing

We welcome contributions to the Data Science Query Assistant! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request.

## License

[MIT](LICENSE.md) - Feel free to use, modify, and distribute this software as you see fit.

## Acknowledgements

- Thanks to Cohere and EdenAI for providing the AI models used in this project.
- This project was created using Streamlit, an open-source app framework for Machine Learning and Data Science projects.
