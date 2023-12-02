# Python Tkinter LLM Playground
I know it's ugly, that's the point.
## Description

This Python application offers a straightforward and simple gui for accessing completions 
APIs compatible with OpenAI or LlamaCpp. Leveraging Tkinter, which is included in all 
Python distributions, it provides an easy-to-use environment with minimal learning curve. 
The application is designed with simplicity in mind, ensuring accessibility for users of 
varying technical backgrounds. The only external dependency required is the `requests` 
library, emphasizing the tool's lightweight and straightforward nature.

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
      "example_template": "template_string"
    },
    // Add more models as needed
  ]
}
```

#### Each model entry must specify:

* **api_type**: The type of the API, e.g., openai_completions or llamacpp_completions.
* **name**: A unique name for the model.
* **api_key**: The API key for authentication (if required).
* **api_url**: The URL endpoint for the model's API.
* **example_template**: A template string to pre-populate the input field.

#### Configuring the Application

Ensure you have the configuration file prepared with the necessary model details.
Place the configuration file in the same directory as the application or provide
the path when running the application.

## Usage

### Requirements

Insure requests is installed.

```bash
pip3 install requests
```

### Running
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
