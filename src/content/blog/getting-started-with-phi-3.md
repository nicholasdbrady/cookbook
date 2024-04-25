---
title: 'Deploy and demo Phi-3 in Azure AI'
description: 'Here is a sample of some basic Markdown syntax that can be used when writing Markdown content in Astro.'
pubDate: 'Apr 24 2024'
heroImage: '/phi-3-hero.png'
---

In this walkthrough, I'll show you how to get started with Phi-3-Mini with the Azure AI Model Catalog. We will deploy the model to an endpoint, test the endpoint, and build a simple chat interface using Gradio.

[comment]: <> (Add a table of contents with referenceable links to click to each section in the blog.)

#### Table of Contents
 - [Introduction](#introduction)
 - [What is Phi-3](#what-is-phi-3)
 - [Use cases](#use-cases)
 - [Getting started](#getting-started-with-phi-3)
 - [Call an ednpoint](#call-an-azure-online-managed-endpoint)
 - [Build a Chat Interface using Gradio](#building-a-chatbot-interface-with-phi-3)
 - [Conclusion](#conclusion)

## Introduction

Artificial Intelligence (AI) has taken great strides in recent years, especially with the development of Large Language Models (LLMs) that have transformed how we interact with technology. However, the computational demands of LLMs can be a barrier for some organizations. Enter Phi-3, the new player in the game. Phi-3 pushes the boundaries by offering powerful language understanding in a compact package—think big AI capabilities with small AI agility.

### What is Phi-3?  
  
Phi-3 is Microsoft's latest innovation—a family of small language models (SLMs) that mark a revolution in AI accessibility and functionality. These models, which are significantly smaller than conventional LLMs, do not compromise on performance. In fact, the smallest member, Phi-3-mini, boasts 3.8 billion parameters and outperforms models twice its size. As more sophisticated versions, Phi-3-small and Phi-3-medium, are set to join the family, developers have more choices to tailor AI capabilities to their needs without the heavy resource requirements.  
  
![Placeholder for Phi-3 overview diagram]

### Use Cases  
- **Local Operation**: Deploy AI in remote areas or on devices with limited connectivity.  
- **Edge Computing**: Integrate AI into smart devices for real-time processing.  
- **Specific Tasks**: Tailor models for targeted tasks like summarizing documents, aiding creative writing, or powering chatbots.  
  
### Getting Started with Phi-3  
  
#### Step 1: Set Up Your Azure Account  
Before you dive into using Phi-3, you'll need to set up an Azure account if you don't already have one. Visit the Azure website and follow the sign-up instructions. 

#### Step 2: Access the Azure AI Model Catalog
Once your account is set up, navigate to the Azure AI Model Catalog where you'll find the Phi-3 models listed.
[Link to Azure AI Model Catalog](https://azure.com/ai-model-catalog)  

![Placeholder for a gif on navigating the Azure AI Model Catalog]

#### Step 3: Choose Your Phi-3 Model
Decide which Phi-3 model suits your requirements. As a beginner, consider starting with Phi-3-mini due to its smaller size and ease of use.

#### Step 4: Deploying the Model
Follow the Azure AI guidelines to deploy the Phi-3 model on your preferred platform. Whether you're running models locally on Ollama or using Azure's powerful cloud computing, you'll find clear instructions to get started.
[Link to Azure AI deployment instructions](https://azure.com/ai-deployment-instructions)

![Placeholder for a video on deploying Phi-3]

### Call an Azure Online Managed Endpoint  
Instead of fine-tuning the model, which can be complex for beginners, you might want to immediately start using the Phi-3 model through an Azure Managed Online Endpoint. Azure Managed Online Endpoints allow you to deploy your models as a web service easily, so you can send data to your model and receive predictions in return.  
  
Here's a simple walkthrough on how you call the Phi-3 Online Managed Endpoint:  
  
#### Make sure you have the following prerequisites:  
- An Azure Machine Learning workspace  
- The `azureml-core` Python package installed  
- An instance of the Phi-3 model deployed to an Online Endpoint  
  
#### Using `phi-3.py` code:  
```python  
from azureml.core.workspace import Workspace  
from azureml.core.model import Model  
from azureml.core import Webservice  
  
# Replace with your Azure subscription details  
subscription_id = "<YOUR_SUBSCRIPTION_ID>"  
resource_group = "<YOUR_RESOURCE_GROUP>"  
workspace_name = "<YOUR_WORKSPACE_NAME>"  
  
# Authenticate to Azure ML Workspace  
workspace = Workspace(subscription_id, resource_group, workspace_name)  
  
# Replace with your model and online endpoint names  
model_name = "Phi-3-mini"  
endpoint_name = "<ENDPOINT_NAME>"  
  
# Access the endpoint  
service = Webservice(workspace, endpoint_name)  
  
# Prepare the data payload  
data = {  
    "Inputs": {  
        "data": [  
            "The data you want the model to process"  
        ],  
    },  
    "GlobalParameters": {}  
}  
  
# Make a call to the endpoint  
predictions = service.run(input_data=json.dumps(data))  
  
print(f"Received predictions: {predictions}")
```

#### Integrate and Test
Once your model is up and running, integrate it into your application and carry out thorough testing to make sure it fulfills your expectations.

![Placeholder for a gif showing integration and testing process]

#### Building a Chatbot Interface with Phi-3  
  
After getting familiar with the Phi-3 model, let's put it into action by creating a chatbot interface. We'll be using Gradio, a Python library that allows you to rapidly create UIs for your machine learning models.  
  
##### Prerequisites  
Before starting, make sure you have:  
- Python installed on your system.  
- An Azure account with Phi-3 deployed (as explained in previous sections).  
  
##### Step 1: Install Gradio  
If you haven't already, install Gradio by running the following command in your terminal:  
  
```bash  
pip install gradio  
``` 

##### Step 2: Create Your Chatbot Logic
You'll need to write a function that connects to your Phi-3 model and processes the user input. Here's an example:

```python
import gradio as gr  
from your_model_library import phi3_interface  # Replace with your actual import  
  
def chatbot_function(message):  
    response = phi3_interface.query(message)  # Replace with actual method to call Phi-3  
    return response  
  
# Test the function to ensure it's working correctly  
print(chatbot_function("Hello, Phi-3!"))  # Expected: Phi-3's greeting response  
``` 

##### Step 3: Set Up Gradio Interface
Now, let's set up a simple chat interface with Gradio:
```python
iface = gr.Interface(  
    fn=chatbot_function,  
    inputs=gr.inputs.Textbox(lines=2, placeholder="Type a message..."),  
    outputs="text",  
    title="Phi-3 Chatbot Example",  
    description="A simple chatbot interface using Phi-3 model."  
)  
  
# Launch the interface  
iface.launch()  
```

##### Step 4: Interact with Your Chatbot
Once you run the code above, Gradio will generate a link to a web interface where you can interact with your chatbot. Type in your messages and see how the Phi-3 model responds!

![Placeholder for Gradio interface screenshot]

### Conclusion 
Throughout this blog, we've embarked on an exciting journey into the world of small language models with Microsoft's Phi-3 series. We've explored its impressive performance, understanding its potential to democratize AI by bringing advanced AI capabilities within reach of more developers and organizations. By starting with the introduction of Phi-3, highlighting its efficiency, adaptability, and user-friendliness, we've set the stage for practical application.  
  
Taking the leap from concept to practice, we walked through the process of setting up an Azure account, deploying the Phi-3 model, and calling an Azure Online Managed Endpoint for seamless AI integration. We topped off our adventure by getting our hands dirty with some code, crafting a chatbot using the powerful yet accessible Phi-3-mini model and Gradio, thus demonstrating the model's versatility and ease of use.  
  
As we wrap up, we can reflect on the key takeaways:  
- **Phi-3 models bridge the gap** between the need for powerful AI and the constraints of computational resources.  
- **Step-by-step guidance bolsters confidence** in deploying and integrating AI into applications.  
- **Practical exercises solidify understanding**, as experienced with our own Gradio-powered chatbot.  
  
Now that you're equipped with the knowledge and tools to harness Phi-3, it's time to dive deeper. Experiment with the models, fine-tune them to your requirements, and conceive creative applications that push the boundaries of what small language models can do.  
  
To keep the momentum going, consider the following actions:  
- [Share your Gradio chatbot](#) and swap insights with peers.  
- [Explore further AI models in Azure AI](https://azure.com/ai-models) to discover other gems.  
- [Join the Azure AI developer community](https://azure.com/ai-community) for support and inspiration.  
  
Don't stop here—continue exploring, learning, and building. Your next breakthrough in AI is just around the corner, and we can't wait to see what you create with Azure AI and the Phi-3 series.  

[Return to top](#table-of-contents)
  
[Placeholder for a vibrant call-to-action graphic encouraging readers to engage with the community]