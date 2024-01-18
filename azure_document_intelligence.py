# Imports
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import pandas as pd
import functools

# only for development
import os
from dotenv import load_dotenv
load_dotenv()

################### DEFINE PARAMETERS ###################
# azure credentials
ocr_key = os.getenv('OCR_KEY')
ocr_endpoint = os.getenv('OCR_ENDPOINT')

# general
ocr_type = 'text'       # type of OCR: text, form, table
locale = 'en-US'        # optional, language of the document
file_list = ''          # dataframe containing the file names
path_column = ''        # column that contains the file path

n_con_retry = 3         # number of retries if connection fails
retry_delay = 2         # delay between retries

# for text extraction
lod = 'line'            # level of detail: word, line, paragraph, page

# for form extraction

# for table extraction


##################### HELPER FUNCTIONS #####################
def format_azure_polygon(polygon):
    """ Format the Azure polygon output into a list of coordinates 
    
    Args:
    ----
        polygon: list of coordinates in Azure format
    
    Returns:
    -------
        polygon: list of coordinates in standard format
    """
    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    x3, y3 = polygon[2]
    x4, y4 = polygon[3]

    return x1, y1, x2, y2, x3, y3, x4, y4

def retry_on_endpoint_connection_error(max_retries=3, delay=2):
    """
    This is a decorator function that allows a function to retry execution when an EndpointConnectionError occurs.

    Parameters:
    max_retries (int): The maximum number of retries if an EndpointConnectionError occurs. Default is 3.
    delay (int): The delay (in seconds) between retries. Default is 2.

    Returns:
    wrapper function: The decorated function that includes retry logic.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                    """ except EndpointConnectionError as e:
                        SAS.logMessage(f'Retrying due to EndpointConnectionError: {e}')
                        retries += 1
                        time.sleep(delay) """
                except Exception as e:
                    raise e  # Let other exceptions be handled by the utility class

            if retries == max_retries:
                #SAS.logMessage(f"Max retries ({max_retries}) reached. Unable to complete operation.", 'warning')
                raise RuntimeError("Max retries to contact Azure endpoint reached. Unable to complete operation.")
        return wrapper

    return decorator

###################### OCR STRATEGIES #####################
# parent class for the OCR strategies
class OCRStrategy:
    """ Base class for the OCR strategies """
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.kwargs = kwargs

    def parse_ocr_result(self) -> pd.DataFrame:
        pass

    def analyze_document(self, document) -> pd.DataFrame:
        pass

# implemented OCR strategies
class ExtractText(OCRStrategy): 
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.lod = kwargs.get('lod', 'line')
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', 'en-US')

    def parse_ocr_result(self, result, lod) -> pd.DataFrame:
        """ Parse the OCR result and return a dataframe
        
        Parameters:
        -----------
        result:
            Azure OCR result
        lod:
            str: level of detail (word, line, paragraph, page)
        
        Returns:
        --------
        df:
            pd.DataFrame: table with parsed OCR results """

        df = pd.DataFrame() 

        # azure doesn't provide results on page level natively
        level = 'LINE' if lod.upper() == 'PAGE' else lod.upper()

        # iterate over all pages in the result
        for page in result.pages:
            if result.styles:
                contains_handwriting = getattr(result.styles[0], 'is_handwritten', False)
            else:
                contains_handwriting = False
            parsed_result = []
            
            # calculating metrics 
            if level != 'WORD':
                word_confidences = [word.confidence for word in page.words]
                total_confidence = sum(word_confidences)
                total_words = len(word_confidences)
                average_confidence = total_confidence / total_words if total_words > 0 else 0
                
            # extraction of (natively provided) results 
            if level == 'PARAGRAPH':
                for paragraph_idx, paragraph in enumerate(result.paragraphs):
                    x1, y1, x2, y2, x3, y3, x4, y4 = format_azure_polygon(paragraph.bounding_regions[0].polygon)

                    paragrpah_info = {
                        'page': paragraph.bounding_regions[0].page_number,
                        'line': paragraph_idx,
                        'text': paragraph.content,
                        'role': paragraph.role,
                        'bb_x1': x1,
                        'bb_y1': y1,
                        'bb_x2': x2,
                        'bb_y2': y2,
                        'bb_x3': x3,
                        'bb_y3': y3,
                        'bb_x4': x4,
                        'bb_y4': y4,
                    }
                    
                    parsed_result.append(paragrpah_info)

            elif level == 'LINE':
                for line_idx, line in enumerate(page.lines):
                    x1, y1, x2, y2, x3, y3, x4, y4 = format_azure_polygon(line.polygon)

                    line_info = {
                        'page': page.page_number,
                        'line': line_idx,
                        'text': line.content,
                        'bb_x1': x1,
                        'bb_y1': y1,
                        'bb_x2': x2,
                        'bb_y2': y2,
                        'bb_x3': x3,
                        'bb_y3': y3,
                        'bb_x4': x4,
                        'bb_y4': y4,
                    }
                    
                    parsed_result.append(line_info)

            elif level == "WORD":
                for word in page.words:
                    x1, y1, x2, y2, x3, y3, x4, y4 = format_azure_polygon(word.polygon)

                    word_info = {
                        'page': page.page_number,
                        'text': word.content,
                        'confidence': word.confidence,
                        'bb_x1': x1,
                        'bb_y1': y1,
                        'bb_x2': x2,
                        'bb_y2': y2,
                        'bb_x3': x3,
                        'bb_y3': y3,
                        'bb_x4': x4,
                        'bb_y4': y4,
                    }
                    
                    parsed_result.append(word_info)
            
            # page level aggregates line level results
            if lod.upper() == 'PAGE':
                page_df = pd.DataFrame(parsed_result)
                parsed_result = []
                page_info = {
                    'page': page.page_number,
                    'text': '\n '.join(page_df['text']),
                    'confidence': average_confidence,
                    'contains_handwriting': contains_handwriting,
                    'bb_x1': df['bb_x1'].min(),
                    'bb_y1': df['bb_y1'].min(),
                    'bb_x2': df['bb_x2'].max(),
                    'bb_y2': df['bb_y2'].min(),
                    'bb_x3': df['bb_x3'].max(),
                    'bb_y3': df['bb_y3'].max(),
                    'bb_x4': df['bb_x4'].min(),
                    'bb_x4': df['bb_x4'].max(),
                }
                parsed_result.append(page_info)
                
            df = pd.concat([df, pd.DataFrame(parsed_result)])
            

        return df  

    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> pd.DataFrame:
        """ Analyze the document and return the result
        
        Returns:
        --------
        parsed_result:
            pd.DataFrame: OCR results
         """
        poller = self.ocr_client.begin_analyze_document('prebuilt-layout', 
                                                         document = document, 
                                                         locale = self.locale)
        result = poller.result()

        parsed_result = self.parse_ocr_result(result, self.lod)
        
        return parsed_result

class ExtractForm(OCRStrategy):
    def __init__(self, ocr_client, kwargs):
        pass

    def parse_ocr_result(self, document) -> pd.DataFrame:
        pass

    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self) -> pd.DataFrame:
        pass

class ExtractTable(OCRStrategy):
    def __init__(self, ocr_client, kwargs): 
        pass

    def parse_ocr_result(self, document) -> pd.DataFrame:
        pass
    
    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self) -> pd.DataFrame:
        pass

# class that processes the OCR
class OCRProcessor:
    """ Class that processes the OCR depending on the strategy"""
    def __init__(self, ocr_client: DocumentAnalysisClient, ocr_type:str, **kwargs):
        self.ocr_client = ocr_client
        self.ocr_type = ocr_type
        self.kwargs = kwargs

        # Define the strategy mapping
        self.strategy_mapping = {
            ('text'): ExtractText,
            ('form'): ExtractForm,
            ('table'): ExtractTable
        }

        # Get the strategy class, parameters and initiate strategy
        strategy_class = self.strategy_mapping[(self.ocr_type)]
        self.strategy = strategy_class(ocr_client = self.ocr_client, kwargs = self.kwargs)

    def analyze_document(self, document):
        return self.strategy.analyze_document(document)
    
###################### EXECUTION ######################
# test input (for dev only)
data = {'file_path': ['data/handwritten-form.jpg','data/letter-example.pdf'],
        'filename': ['doc1', 'doc2']}

file_list = pd.DataFrame(data)
path_column = 'file_path'


# define the parameters for the OCR
ocr_params = {
              'ocr_level': lod,
              'locale': locale,
              }

# initiate the OCR client and processor
ocr_client = DocumentAnalysisClient(endpoint = ocr_endpoint, 
                                    credential=AzureKeyCredential(ocr_key)
                                    )

ocr_processor = OCRProcessor(ocr_client = ocr_client, 
                             ocr_type = ocr_type, 
                             **ocr_params
                             )

# initiate dataframe to store results and status
ocr_results = pd.DataFrame()
status = pd.DataFrame()

# go through every document in the list
for _, row in file_list.iterrows():
    print(f'processing file {row[path_column]}')
    done = False
    error_type = ''
    message = ''

    # read the document
    with open(row[path_column], 'rb') as document:
        document = bytearray(document.read())

    # analyze the document
    try:
        result = ocr_processor.analyze_document(document)
        ocr_results = pd.concat([ocr_results, result], ignore_index=True)
        ocr_results[path_column]=row[path_column]
        done = True
    except Exception as e:
        error_type = type(e).__name__
        message = str(e)
        print(f'Error: {error_type} - {message}')
    
    # update the status
    doc_status = {'file': row[path_column],
                  'done': done,
                  'error_type': error_type,
                  'message': message
                  }
    
    status = pd.concat([status, pd.DataFrame(doc_status, index=[0])], ignore_index=True)


# print & save the results (dev only)
print(f'Successfully processed {status["done"].sum()} / {status.shape[0]} files!')
ocr_results.to_csv('ocr_results.csv')
status.to_csv('ocr_status.csv')


    
