import datetime
import tkinter as tk
from tkinter import ttk
import requests
import json
import threading
from typing import Iterable, Dict


class ModelLlamaCpp:
    """
    A class representing the ModelLlamaCpp for generating text completions.

    This class provides methods to stream responses from the LlamaCpp API based on a given prompt.
    It supports streaming with callbacks and can be controlled to start and stop the stream as needed.

    Attributes:
        name (str): The name of the model.
        api_key (str): The API key for authentication.
        api_url (str): The URL for the API endpoint.
        example_template (str): An example template for the model.
        thread (threading.Thread): The thread for streaming data.
        stop_requested (bool): Flag to indicate if streaming should be stopped.
    """

    name: str
    api_key: str
    api_url: str
    example_template: str

    def __init__(self, name: str, api_key: str, api_url: str, example_template: str = ""):
        """
        Initializes the ModelLlamaCpp instance.

        Args:
            name (str): The name of the model.
            api_key (str): The API key for authentication.
            api_url (str): The URL for the API endpoint.
            example_template (str, optional): An example template for the model. Defaults to an empty string.
        """
        self.thread = None
        self.name = name
        self.api_key = api_key
        self.api_url = api_url
        self.example_template = example_template
        self.stop_requested = False

    def stream(self, prompt: str, max_tokens: int = 100, temperature: float = 0) -> Iterable[str]:
        """
        Streams the response from the model based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Controls randomness in the generation. Defaults to 0.

        Yields:
            Iterable[str]: An iterable of the model's text completions.
        """
        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            'model': self.name, 'prompt': prompt, 'n_predict': max_tokens,
            'temperature': temperature, 'stream': True
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data), stream=True)
        for line in response.iter_lines():
            print(line)
            # Check if the line is not empty and not the end-of-stream marker
            if line and not line.startswith(b'data: [DONE]'):
                decoded_line = line.decode('utf-8')

                # The line starts with "data: ", so remove that part to get the JSON
                json_data = json.loads(decoded_line[6:])  # Skipping the first 6 characters "data: "

                if 'content' in json_data:
                    # Extract and yield the text from the first choice
                    text = json_data['content']
                    if text:
                        yield text

    def stream_with_callback(self, prompt, callback, done, max_tokens: int = 100, temperature: float = 0):
        """
        Initiates streaming with a callback function for each completion and a done function after completion.

        Args:
            prompt (str): The input prompt for the model.
            callback (function): A callback function that will be called with each completion.
            done (function): A function that will be called when streaming is done.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Controls randomness in the generation. Defaults to 0.
        """
        self.stop_requested = False  # Reset the flag

        def stream_thread():
            for completion in self.stream(prompt, max_tokens, temperature):
                if self.stop_requested:  # Check if stop is requested
                    break
                callback(completion)
            done()

        self.thread = threading.Thread(target=stream_thread)
        self.thread.start()


class ModelOpenAI:
    """
    A class representing the OpenAI model for generating text completions.

    Attributes:
        name (str): The name of the model.
        api_key (str): The API key for authentication.
        api_url (str): The URL for the API endpoint.
        example_template (str): An example template for the model.
        thread (threading.Thread): The thread for streaming data.
        stop_requested (bool): Flag to indicate if streaming should be stopped.
    """

    name: str
    api_key: str
    api_url: str
    example_template: str

    def __init__(self, name: str, api_key: str, api_url: str, example_template: str = ""):
        """
        Initializes the ModelOpenAI instance.

        Args:
            name (str): The name of the model.
            api_key (str): The API key for authentication.
            api_url (str): The URL for the API endpoint.
            example_template (str, optional): An example template for the model. Defaults to an empty string.
        """
        self.thread = None
        self.name = name
        self.api_key = api_key
        self.api_url = api_url
        self.example_template = example_template
        self.stop_requested = False

    def stream(self, prompt: str, max_tokens: int = 100, temperature: float = 0) -> Iterable[str]:
        """
        Streams the response from the model based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Controls randomness in the generation. Defaults to 0.

        Yields:
            Iterable[str]: An iterable of the model's text completions.
        """
        headers = {
            'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'model': self.name, 'prompt': prompt, 'max_tokens': max_tokens,
            'temperature': temperature, 'stream': True
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data), stream=True)
        for line in response.iter_lines():
            print(line)
            # Check if the line is not empty and not the end-of-stream marker
            if line and not line.startswith(b'data: [DONE]'):
                decoded_line = line.decode('utf-8')

                # The line starts with "data: ", so remove that part to get the JSON
                json_data = json.loads(decoded_line[6:])  # Skipping the first 6 characters "data: "

                if 'choices' in json_data and len(json_data['choices']) > 0:
                    # Extract and yield the text from the first choice
                    text = json_data['choices'][0].get('text', '')
                    if text:
                        yield text

    def stream_with_callback(self, prompt, callback, done, max_tokens: int = 100, temperature: float = 0):
        """
        Initiates streaming with a callback function for each completion and a done function after completion.

        Args:
            prompt (str): The input prompt for the model.
            callback (function): A callback function that will be called with each completion.
            done (function): A function that will be called when streaming is done.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Controls randomness in the generation. Defaults to 0.
        """
        self.stop_requested = False  # Reset the flag

        def stream_thread():
            for completion in self.stream(prompt, max_tokens, temperature):
                if self.stop_requested:  # Check if stop is requested
                    break
                callback(completion)
            done()

        self.thread = threading.Thread(target=stream_thread)
        self.thread.start()


def select_all(event):
    event.widget.tag_add('sel', '1.0', 'end')
    return 'break'


def load_config(file_path: str) -> Dict[str, ModelOpenAI]:
    """
    Loads the configuration from a JSON file and creates model instances.

    Args:
        file_path (str): The path to the configuration file.

    Returns:
        Dict[str, ModelOpenAI]: A dictionary of model name to ModelOpenAI instance.
    """

    with open(file_path, "r") as f:
        cfg = json.load(f)  # Directly use json.load

    if not isinstance(cfg, dict):
        raise ValueError("Invalid config format: not a dictionary.")

    if 'models' not in cfg:
        raise ValueError("No models in config.")

    def create_model(model_config):
        if 'name' not in model_config:
            raise ValueError("Model missing name.")
        if 'api_key' not in model_config:
            raise ValueError(f"Model {model_config['name']} missing api_key.")
        if 'api_url' not in model_config:
            raise ValueError(f"Model {model_config['name']} missing api_url.")

        if 'api_type' in model_config and model_config['api_type'] == "llamacpp_completions":
            return ModelLlamaCpp(
                name=model_config['name'],
                api_key=model_config['api_key'],
                api_url=model_config['api_url'],
                example_template=model_config.get('example_template', "")
            )
        else:
            return ModelOpenAI(
                name=model_config['name'],
                api_key=model_config['api_key'],
                api_url=model_config['api_url'],
                example_template=model_config.get('example_template', "")
            )

    return {model_config['name']: create_model(model_config) for model_config in cfg['models']}


class App:
    """
    Main application class for the chatbot interface built using Tkinter.

    This class sets up the GUI for a chatbot application, handling configuration,
    user input, model selection, and display of the chat history.

    Attributes:
        models (dict): A dictionary mapping model names to their respective instances.
        config_fields (dict): Configuration fields for the application (e.g., temperature, max_tokens).
        chat_history (tk.Text): Text widget for displaying chat history.
        input_field (tk.Text): Text widget for user input.
        model_combobox (ttk.Combobox): Combobox widget for selecting the chat model.
        submit_button (tk.Button): Button widget to submit the user input.
        clear_chat_button (tk.Button): Button widget to clear the chat history.
        stop_button (tk.Button): Button widget to stop the ongoing chat generation.
    """

    def __init__(self, config_file: str):
        """
        Initializes the application with the given configuration file.

        Args:
            config_file (str): The path to the configuration file.
        """
        self.models = load_config(config_file)
        self.config_fields = {'temperature': 0.7, 'max_tokens': 1024}

        root = tk.Tk()
        root.title("Chatbot")
        root.geometry('1100x600')

        # Main PaneWindow divided into left and right frames
        main_pane = tk.PanedWindow(root, orient='horizontal', sashrelief='raised', sashwidth=4)
        main_pane.pack(fill='both', expand=True)

        # Left pane for chat history and input field
        left_pane = tk.PanedWindow(main_pane, orient='vertical', sashrelief='raised', sashwidth=4)
        main_pane.add(left_pane, width=900)  # Allocate width to left pane

        # Chat history frame within the left pane
        chat_frame = tk.Frame(left_pane)
        left_pane.add(chat_frame, height=400)  # Allocate more space to chat history

        self.chat_history = tk.Text(chat_frame, bd=0, bg='white', font='Arial')
        self.chat_history.pack(side='left', fill='both', expand=True)
        self.chat_history.config(state='disabled')

        # Configure tags for colors
        self.chat_history.tag_config('ai', foreground='black')
        self.chat_history.tag_config('user', foreground='RoyalBlue')

        # Scrollbar for chat history
        scrollbar = tk.Scrollbar(chat_frame, command=self.chat_history.yview)
        scrollbar.pack(side='right', fill='y')
        self.chat_history['yscrollcommand'] = scrollbar.set

        # Input frame (resizable) within the left pane
        input_frame = tk.Frame(left_pane)
        left_pane.add(input_frame, height=200)  # Allocate less space to input field

        # Input field (Text widget for multi-line input) and scrollbar
        input_field_frame = tk.Frame(input_frame)
        input_field_frame.pack(side='left', fill='both', expand=True)
        input_field_frame.grid_propagate(False)

        self.input_field = tk.Text(input_field_frame, bd=0, bg='white', font='Arial')
        self.input_field.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        self.input_field.bind("<Control-Return>", self.on_submit)
        self.input_field.bind('<Control-a>', select_all)
        self.input_field.bind('<Control-A>', select_all)

        # Scrollbar for input field
        input_scrollbar = tk.Scrollbar(input_field_frame, command=self.input_field.yview)
        input_scrollbar.pack(side='right', fill='y')
        self.input_field['yscrollcommand'] = input_scrollbar.set

        # Right sidebar for configuration and submit button
        right_sidebar = tk.Frame(main_pane, bd=2, relief='sunken', padx=5, pady=5)
        main_pane.add(right_sidebar, width=200)  # Allocate width to right sidebar

        # Dropdown for Model selection
        model_label = tk.Label(right_sidebar, text='Model')
        model_label.pack(side='top', fill='x', pady=2)

        # Create a Combobox for models
        self.model_combobox = ttk.Combobox(right_sidebar, values=list(self.models.keys()))
        self.model_combobox.pack(side='top', fill='x', pady=2)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_change)
        if self.models:  # Set the default value if models are available
            self.model_combobox.set(next(iter(self.models)))
        self.on_model_change()

        # Configuration fields for the rest of the entries
        self.config_entries = {}

        for label, default_value in self.config_fields.items():
            frame = tk.Frame(right_sidebar)
            frame.pack(fill='x', pady=2)
            tk.Label(frame, text=label).pack(side='left')

            # Create Entry widget with default value
            entry = tk.Entry(frame)
            entry.pack(side='right', fill='x', expand=True)
            entry.insert(0, str(default_value))  # Insert default value

            # Store the Entry widget in the dictionary
            self.config_entries[label] = entry

        # Submit button within the right sidebar
        self.submit_button = tk.Button(right_sidebar, text="Send", width='12', height=2, bg='#FFFFFF', bd=0,
                                       command=self.on_submit)
        self.submit_button.pack(side='top', pady=10)  # Place the button at the top of the sidebar

        # Clear chat history button within the right sidebar
        self.clear_chat_button = tk.Button(right_sidebar, text="Clear Chat", width='12', height=2, bg='#FFFFFF', bd=0,
                                      command=self.clear_chat_history)
        self.clear_chat_button.pack(side='top', pady=10)  # Place the button under the submit button

        self.stop_button = tk.Button(right_sidebar, text="Stop", width='12', height=2, bg='#FFFFFF', bd=0,
                                     command=self.on_stop, state='disabled')
        self.stop_button.pack(side='top', pady=10)

        root.mainloop()

    def on_model_change(self, event=None):
        """
        Handles the event when the model selection is changed in the GUI.

        Updates the input field with the example template of the selected model.

        Args:
            event: The event triggered when the model is changed. Defaults to None.
        """
        # Get the selected model name
        selected_model_name = self.model_combobox.get()
        model = self.models.get(selected_model_name)

        # Update the input field with the example template of the selected model
        if model and model.example_template:
            self.input_field.delete("1.0", 'end')  # Clear existing text
            self.input_field.insert("1.0", model.example_template)

    def update_chat_history(self, completion: str):
        """
        Updates the chat history with the model's completion.

        Args:
            completion (str): The text completion generated by the model.
        """
        self.chat_history.config(state='normal')
        self.chat_history.insert('end', completion, 'ai')
        self.chat_history.config(state='disabled')
        self.chat_history.see('end')

    def update_chat_history_complete(self):
        """
        Updates the chat history and button states upon completion of the generation process.
        """
        # Update the button states on completion
        self.update_button_states(submit_enabled=True, stop_enabled=False, clear_enabled=True)

    def on_stop(self):
        """
        Handles the event when the 'Stop' button is pressed.

        Signals the model to stop generating text and updates the button states.
        """
        selected_model_name = self.model_combobox.get()
        model = self.models.get(selected_model_name)
        if model:
            model.stop_requested = True  # Signal the thread to stop

        # Update the button states immediately
        self.update_button_states(submit_enabled=True, stop_enabled=False, clear_enabled=True)

        # Start a periodic check to see if the thread has stopped
        self.check_thread_stopped(model)

    def check_thread_stopped(self, model):
        """
        Periodically checks if the model's streaming thread has stopped.

        Args:
            model (ModelOpenAI or ModelLlamaCpp): The model instance whose thread is being checked.
        """
        if model.thread and model.thread.is_alive():
            # If the thread is still alive, check again after a short delay
            self.chat_history.after(100, self.check_thread_stopped, model)
        else:
            # If the thread has stopped, update the GUI as needed
            self.update_chat_history_complete()

    def on_submit(self, event=None):
        """
        Handles the event when the 'Submit' button is pressed or enter key is pressed.

        Sends the user input to the model for generating completions.

        Args:
            event: The event triggered by pressing the 'Submit' button or enter key. Defaults to None.
        """
        user_input = self.input_field.get("1.0", 'end-1c')  # Get text from Text widget
        if user_input.strip():  # Check if input is not just whitespace
            # Disable the input field and submit button
            self.update_button_states(submit_enabled=False, stop_enabled=True, clear_enabled=False)

            self.chat_history.config(state='normal')
            # self.chat_history.insert('end', 'User: ' + user_input + '\n', 'user')

            self.chat_history.insert('end', f"\n\nGeneration started {datetime.datetime.now()}:\n\n", 'user')
            max_tokens = int(self.config_entries['max_tokens'].get())
            temperature = float(self.config_entries['temperature'].get())
            selected_model_name = self.model_combobox.get()
            model = self.models.get(selected_model_name)
            model.stream_with_callback(
                user_input,
                lambda completion: self.chat_history.after(0, self.update_chat_history, completion),
                self.update_chat_history_complete,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.stop_button.config(state='normal')

            # self.input_field.delete("1.0", 'end')  # Clear the Text widget
            self.chat_history.config(state='disabled')
            self.chat_history.see('end')
        return 'break'

    def update_button_states(self, submit_enabled=True, stop_enabled=False, clear_enabled=True):
        """
         Updates the states (enabled/disabled) of the buttons in the GUI.

         Args:
             submit_enabled (bool, optional): Enable or disable the 'Submit' button. Defaults to True.
             stop_enabled (bool, optional): Enable or disable the 'Stop' button. Defaults to False.
             clear_enabled (bool, optional): Enable or disable the 'Clear Chat' button. Defaults to True.
         """
        if submit_enabled:
            self.submit_button.config(state='normal')
        else:
            self.submit_button.config(state='disabled')

        if stop_enabled:
            self.stop_button.config(state='normal')
        else:
            self.stop_button.config(state='disabled')

        if clear_enabled:
            self.clear_chat_button.config(state='normal')
        else:
            self.clear_chat_button.config(state='disabled')

    def clear_chat_history(self):
        """
        Clears the chat history in the GUI.
        """
        self.chat_history.config(state='normal')
        self.chat_history.delete("1.0", 'end')
        self.chat_history.config(state='disabled')


if __name__ == '__main__':
    App("config.json")
