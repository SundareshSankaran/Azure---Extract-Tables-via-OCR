# How-To Guide: Installing a Local Azure Document Intelligence Container

This guide provides detailed steps on how to install a local Azure Document Intelligence container. Ensure you have the necessary prerequisites like Azure account, Docker, and appropriate permissions. [Click here](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-3.0.0&preserve-view=true&tabs=read#billing) to find the original documentation.

> NOTE: So far only Azure Document Intelligence V3.0 (`22-08-31`) is supported. For more info [click here](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-4.0.0&preserve-view=true&tabs=read).

## Contents
- [How-To Guide: Installing a Local Azure Document Intelligence Container](#how-to-guide-installing-a-local-azure-document-intelligence-container)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Run the container with docker-compose up](#run-the-container-with-docker-compose-up)
  - [Validate the service is running](#validate-the-service-is-running)
  - [Stop the container](#stop-the-container)
  - [Further Resources](#further-resources)
  
## Prerequisites
- Azure account
- [Docker engine](https://docs.docker.com/engine/) (or other Docker hosting Service like [Azure Kubernetes Service](https://learn.microsoft.com/en-us/azure/aks/?view=doc-intel-3.0.0))
- [Document Intelligence resource](https://portal.azure.com/#create/Microsoft.CognitiveServicesFormRecognizer)
- [Azure CLI installed](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (optional)
- Host machine with these [requirements](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-3.0.0&preserve-view=true&tabs=read#host-computer-requirements)


## Run the container with docker-compose up
- Replace the `{FORM_RECOGNIZER_ENDPOINT_URI}` and `{FORM_RECOGNIZER_KEY}` values with your resource Endpoint URI and the key from the Azure resource page.
![](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/media/containers/keys-and-endpoint.png?view=doc-intel-3.0.0)

- Ensure that the EULA value is set to accept.
- The EULA, Billing, and ApiKey values must be specified; otherwise the container can't start

> WARNING: Do not share your keys and store them securely.

Create a `docker-compose.yml` file and paste the example code below. Don't forget to replace `{FORM_RECOGNIZER_ENDPOINT_URI}` and `{FORM_RECOGNIZER_KEY}` values for your Layout container instance.\
This code is a docker compose example to run the Document Intelligence **Layout container**. **For other examples [click here!](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-3.0.0&preserve-view=true&tabs=read#run-the-container-with-the-docker-compose-up-command)**
```yml
version: "3.9"
services:
  azure-form-recognizer-read:
    container_name: azure-form-recognizer-read
    image: mcr.microsoft.com/azure-cognitive-services/form-recognizer/read-3.0
    environment:
      - EULA=accept
      - billing={FORM_RECOGNIZER_ENDPOINT_URI}
      - apiKey={FORM_RECOGNIZER_KEY}
    ports:
      - "5000:5000"
    networks:
      - ocrvnet
networks:
  ocrvnet:
    driver: bridge
```
With your terminal at the location of the docker-compose file, execute the following command to start the service.
```bash
docker-compose up
```
For the first launch will pull all required files (~1.7GB). Subsequent executions will just start the service.
## Validate the service is running
There are several ways to validate that the container is running:
- The container provides a homepage at `\` as a visual validation that the container is running.
- You can open your favorite web browser and navigate to the external IP address and exposed port of the container in question. Use the listed request URLs to validate the container is running. The listed example request URLs are `http://localhost:5000`, but your specific container can vary. Keep in mind that you're navigating to your container's **External IP address** and exposed port.

| Request URL                        | Purpose                                                                                                                                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `http://localhost:5000/`           | The container provides a home page.                                                                                                                                                     |
| `http://localhost:5000/ready`      | Requested with GET, this request provides a verification that the container is ready to accept a query against the model. This request can be used for Kubernetes liveness and readiness probes. |
| `http://localhost:5000/status`     | Requested with GET, this request verifies if the api-key used to start the container is valid without causing an endpoint query. This request can be used for Kubernetes liveness and readiness probes. |
| `http://localhost:5000/swagger`    | The container provides a full set of documentation for the endpoints and a Try it out feature. With this feature, you can enter your settings into a web-based HTML form and make the query without having to write any code. After the query returns, an example CURL command is provided to demonstrate the required HTTP headers and body format. |

![](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/media/containers/container-webpage.png?view=doc-intel-3.0.0)

## Stop the container
To stop the containers, use the following command:
```bash
docker-compose down
```

## Further Resources
- [Billing & Billing Arguments](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/install-run?view=doc-intel-3.0.0&preserve-view=true&tabs=read#billing)
- [Configure Document Intelligence containers](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/configuration?view=doc-intel-3.0.0)
- [Deploy and run container on Azure Container Instance](https://learn.microsoft.com/en-us/azure/ai-services/containers/azure-container-instance-recipe?view=doc-intel-3.0.0&tabs=portal)