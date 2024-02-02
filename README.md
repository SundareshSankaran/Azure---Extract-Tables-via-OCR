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

> **Note:** This step works great with the [Create Listings of Directory - CLOD](https://github.com/sassoftware/sas-studio-custom-steps/tree/main/Create%20Listing%20of%20Directory%20CLOD) custom step  to create the input file-list based on a folder of documents. 

### 📺 Tutorial (Click Thumbnail)
[![YOUTUBE THUMBNAIL]()](https://youtu.be/RP0CHuIbVGE)


### Supported File Types

| Model       | PDF | Image[^1] |
|---------------|----------|----------------------------------------|
| Read   | ✅   |  ✅                                               |
| Layout | ✅   | ✅                                      | 

[^1]: JPEG/JPG, PNG, BMP, TIFF, HEIF
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

### General
| Parameter   | Required      | Description                                                      |
|-------------|:---------------:|------------------------------------------------------------------|
| OCR Type    | Yes           | Defines the type of Optical Character Recognition (OCR) to use   |
| Input Mode  | Yes           | Indicates if processing a list of files or a single file         
| File Path   | No*           | The file path for processing a single file                       |
| Input Table | No†           | The name of the table containing file paths/URLs for batch processing |
| Path Column | No†           | The column in the input table that contains the file path/URL    |

\* Required if ``Input Mode`` is set to *"single"*. <br>
† Required if ``Input Mode`` is set to *"batch"*.

<details>
  <summary style="font-size: 16px;">Text Settings</summary>
  
| Parameter            | Required | Description                                                                                     |
|----------------------|:--------:|-------------------------------------------------------------------------------------------------|
| Granularity |    Yes     | Defines granularity of the text output (e.g. word, line, paragrpah, page).  Has implications regarding extraction output (e.g. 'role' only for paragraphs, 'confidence' only for words/pages)<ul><li>**word** -  includes *confidence* value</li><li>**line** - text line per row</li><li>**paragraph** - includes 'role' of a given paragraph (heading, etc..)</li><li>**page** - everything one one page</li></ul> |

</details>

<details>
  <summary style="font-size: 16px;">Query Settings</summary>
  
| Parameter       | Required | Description                                                                                                         |
|-----------------|:--------:|---------------------------------------------------------------------------------------------------------------------|
| Query Fields    |    Yes    | List of keys that are used as queries in the extraction process.                                                    |
| Exclude Metadata|    No    | If set to 'yes', all meta information from the extraction will be ignored, and the output will only contain a column per key and a row per file. |

</details>

<details>
  <summary style="font-size: 16px;">Table Settings</summary>
  
| Parameter             | Required | Description                                                                                                               |
|-----------------------|:--------:|---------------------------------------------------------------------------------------------------------------------------|
| Table Output Format   |    Yes    | Defines the output format for table extraction: <ul><li>**map** - outputs (col_id, row_id, value) for later reconstruction</li><li>**reference** - outputs a row per table with a uuid as reference, stored in the defined library</li><li>**table** - outputs one table through standard output, supports only one table and one file</li></ul> |
| Table Output Library  |    No*    | Defines the output library for extracted. tables                                  |
| Select Tables         |    No†    | Defines if a table per document is selected.                                                          |
| Table Selection Method|    No    | Defines the method to select the table per document that is extracted: <ul><li>**index** - uses the index to select the extracted table.</li><li>**size** - selects the table with the most cells.</li></ul> |
| Table Index           |    No‡    | Table index to extract.                                                    |

\* Only available if ``Table Output Format`` is set to *"reference"*. <br>
† Defaults to true when ``Table Output Format`` is *"table"*. <br>
‡ Required if ``Table Selection Method`` is set to *"index"*

</details>

### 🔐 Azure
| Parameter  | Required | Description |
|---------------------|:----------:|--------------------------------|
|Endpoint| Yes | AI Document Intelligence Resource Endpoint |
|Key |Yes| Secret Key |

<details>
  <summary style="font-weight: bold;">👉Where to find resource key and endpoint</summary>

  ![](img/keys-and-endpoint.png)
</details>
<br>


### 🧙‍♂️ Advanced

| Parameter  | Required | Description |
|---------------------|:----------:|--------------------------------|
|Number of Retries| No |How many retries attempts before a document is skipped|
|Seconds between retries|No| How many seconds between retry attempts|
|Number of Threads|No|How many Python threads will be used to process all files.|
|Save as JSON|No|Whether to save the raw output as JSON (one file per document)|
|Output Folder|No*|Folder for the JSON files.|

\* Required if ``Save as JSON`` is set to *true*.

## 📚 Documentation
- [What is Azure AI Document Intelligence?](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview?view=doc-intel-4.0.0)
- [Azure AI Document Intelligence documentation](https://learn.microsoft.com/en-US/azure/ai-services/document-intelligence/?view=doc-intel-4.0.0&viewFallbackFrom=form-recog-3.0.0&branch=release-build-cogserv-forms-recognizer)
- [Pricing](https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/#pricing)
- [Language Support](https://learn.microsoft.com/en-GB/azure/ai-services/document-intelligence/language-support-ocr?view=doc-intel-4.0.0&tabs=read-print%2Clayout-print%2Cgeneral)
- [Data Privacy](https://learn.microsoft.com/en-us/legal/cognitive-services/document-intelligence/data-privacy-security)
  

## 📝 Change Log
* Version 1.0 (08JAN2024) 
    * Initial version
