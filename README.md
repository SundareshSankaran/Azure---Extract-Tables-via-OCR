# OCR - Azure AI Document Intelligence
This custom step uses the [Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) service to perform different types of [OCR](https://en.wikipedia.org/wiki/Optical_character_recognition) on files that are stored on the SAS file system. [What is Azure AI Document Intelligence?](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview?view=doc-intel-4.0.0)


## ✨ Features
- ✅ Text Extraction (words / lines / paragraphs / pages)
- ✅ Form Extraction (key-value pairs)
- ✅ Query Extraction (extraction of specified keys)
- ✅ Table Extraction
- ✅ Paralel Execution (threading)

## 📖 Contents
- [💻 User Interface](#💻-user-interface)
- [👩‍💻 Usage](#👩‍💻-usage)
- [📋 Requirements](#📋-requirements)
- [⚙️ Settings](#⚙️-settings)
- [📚 Documentation](#📚-documentation)
- [📝 Change Log](#📝-change-log)
## 💻 User Interface

## 👩‍💻 Usage
### 📺 Tutorial (Click Thumbnail)
[![YOUTUBE THUMBNAIL]()](https://youtu.be/RP0CHuIbVGE)



> **Note:** This step works well with the following custom step [Create Listings of Directory - CLOD](https://github.com/sassoftware/sas-studio-custom-steps/tree/main/Create%20Listing%20of%20Directory%20CLOD) to create the input file-list based on a folder of documents. 
> 
### Supported File Types

| Model       | PDF | Image$^1$ | DOCX, XLSX, PPTX, HTML |
|---------------|----------|----------------------------------------|--------------------------------------------------|
| Read   | ✅   | ✅                                     | ✅                                               |
| Layout | ✅   | ✅                                      | ✅                           |

[1] JPEG/JPG, PNG, BMP, TIFF, HEIF
### Test data
- [Sample forms on Github](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_forms)
- [Handwritten Form Sample](https://www.nist.gov/image/sd19jpg)


> **Pro Tip:** Take a photo with your smarphone, make a screenshot of a document or export a PowerPoint slide as image / PDF.


## 📋 Requirements

Tested on SAS Viya version Stable 2024.1

### 🐍 Python
- [Python 3.8](https://www.python.org/downloads/) (or higher)

### 📦 Packages
- [azure.ai.documentintelligence](https://pypi.org/project/azure-ai-documentintelligence/)
- [azure.core](https://pypi.org/project/azure-core/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)

### 🤖 Azure AI Document Intelligence Resource
To use this step the endpoint and key for an Azure Document Intelligence Resource is needed. <br> 👉 [Create a Document Intelligence resource](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0)



## ⚙️ Settings
X = Always Required | O = Required (for certain settings) |  = No user input required
| Parameter           | Required | Type    | Description                                                                                        |
|---------------------|:--------:|---------|----------------------------------------------------------------------------------------------------|
| Extraction Type     | X        | Option  | Specifies the AWS Textract action that is called. For text: DetectDocumentText, for forms: AnalyzeDocument |
| Extraction Level    | X        | Option  | The level of aggregation of the detected text. Possible values: Word, Line, Paragraph, Text       |
| File Location       | X        | Option  | Specifies whether the files are stored locally or in a S3 bucket                                   |
| Input Type          | X        | Option  | Specifies which documents should be processed. For local files (SAS Viya): Single file, list of files (table), For S3 Bucket: list of files (table), or a whole bucket |
| File Path           |O          | Path    | File to be processed. Only when "SAS Viya" and "just one file" is selected                             |
| File List           |O          | Table   | Table containing list of files. Only when a "list of files" is selected                            |
| Document Path Column|O          | Column  | Column that contains the file paths. Only when "list of files" is selected                         |
| S3 Bucket Name      | O        | String  | Name of the S3 bucket containing the files. Only when "S3 Bucket" is selected                      |
| Output Status Table |         | Option  | Whether status tracking information about the processing should be in the output                    |

### 🔐 Azure
| Parameter  | Required | Description |
|---------------------|:----------:|--------------------------------|
|Endpoint| X | Access Key |
|Key |X| Secret Key |

<details>
  <summary>Screenshot</summary>
  
  ![](img/keys-and-endpoint.png)
</details>

### 🧙‍♂️ Advanced
| Parameter  | Required | Description |
|---------------------|:----------:|--------------------------------|
|Number of Retries|  |How many retries attempts before a document is skipped|
|Seconds between retries|| How many seconds between retry attempts|
|Number of Threads||How many Python threads will be used to process all files.|
|Save as JSON||Whether to save the raw output as JSON (one file per document)|
|Output Folder|O|Folder for the JSON files.|


## 📚 Documentation
- [What is Azure AI Document Intelligence?](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview?view=doc-intel-4.0.0)
- [Azure AI Document Intelligence documentation](https://learn.microsoft.com/en-US/azure/ai-services/document-intelligence/?view=doc-intel-4.0.0&viewFallbackFrom=form-recog-3.0.0&branch=release-build-cogserv-forms-recognizer)
- [Pricing](https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/#pricing)
- [Language Support](https://learn.microsoft.com/en-GB/azure/ai-services/document-intelligence/language-support-ocr?view=doc-intel-4.0.0&tabs=read-print%2Clayout-print%2Cgeneral)
  
<details>
  <summary style="font-size: 22px; font-weight: bold;">Screenshot</summary>
  
  | Parameter  | Required | Description |
|---------------------|:----------:|--------------------------------|
|Number of Retries|  |How many retries attempts before a file is skipped.|
|Seconds between retries|| How many seconds between retry attempts.|
|Number of Threads||How many Python threads will be used to process all files.|
|Save as JSON||Whether to save the raw output as JSON (one file per document).|
|Output Folder|O|Folder for the JSON files.|
</details>

## 📝 Change Log
* Version 1.0 (08JAN2024) 
    * Initial version
