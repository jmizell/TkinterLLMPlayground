# Python Tkinter LLM Playground
I know it's ugly, that's the point.
## Description

This Python application utilizes Tkinter, included in standard Python distributions, 
to provide a basic but functional interface for interacting with language model APIs 
compatible with OpenAI and LlamaCpp. The design focuses on simplicity and 
user-friendliness, catering to users with varying levels of technical expertise. There
are no dependencies outside the standard lib. It's primarily intended as a practical 
tool for exploring and testing various open source language models that operate with 
OpenAI or Llama-CPP compatible APIs.

## Configuration

### Config File

The application requires a JSON configuration file to define the models it can interact 
with. An example structure of the config file is as follows: The configuration file
must be called `config.json`.

```json
{
  "models": [
    {
      "api_type": "openai_completions",
      "name": "ModelName1",
      "api_key": "api_key_here",
      "api_url": "http://model.api.url",
      "example_template": "template_string",
      "stop_string": "optional_stop_string",
      "disabled": false
    },
    // Add more models as needed
  ]
}
```

#### Each model entry must specify:

* **api_type**: The type of the API, e.g., openai_chat, openai_completions or llamacpp_completions.
* **name**: A unique name for the model.
* **api_key**: The API key for authentication (if required).
* **api_url**: The URL endpoint for the model's API.
* **example_template**: A template string to pre-populate the input field.
* **stop_string** (optional): A string that, when detected in the output, signals the model to stop generating further text.
* **disabled** (optional): A boolean that disables the model at load time.

#### Configuring the Application

Ensure you have the configuration file prepared with the necessary model details.
Place the configuration file in the same directory as the application or provide
the path when running the application.

## Usage

Start the Application: Run the application by executing the main script. If you have a 
specific configuration file, pass its path as an argument. The configuration file
must be called `config.json`.

```bash
python main_script.py
```

* **Select a Model:** Use the dropdown menu in the application to select the desired AI model.
* **Input and Send Prompts:** Enter your prompt in the input field and press 'Send' or 
  'Ctrl + Enter' to get a response from the selected model.
* **View Responses:** The AI's responses will appear in the chat history area.
* **Adjust Settings:** You can adjust settings like temperature and max_tokens for each model 
  using the provided fields.
* **Stop Generation:** If the model supports streaming, you can stop the generation process 
  anytime using the 'Stop' button.
* **Clear Chat History:** Use the 'Clear Chat' button to clear the chat history.
