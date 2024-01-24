# Imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
import pandas as pd
import functools
import io
import re

# only for development
import os
from dotenv import load_dotenv
load_dotenv()

################### DEFINE PARAMETERS ###################
# azure credentials
ocr_key = os.getenv('OCR_KEY')
ocr_endpoint = os.getenv('OCR_ENDPOINT')

# general
ocr_type = 'query'               # type of OCR: text, form, table, query
locale = 'en-US'                # optional, language of the document
file_list = ''                  # dataframe containing the file names
path_column = ''                # column that contains the file path

n_con_retry = 3                 # number of retries if connection fails
retry_delay = 2                 # delay between retries

save_json = False               # whether to save the json output
output_folder = ''              # folder to save the json output  

# for text extraction
lod = 'line'                    # level of detail: word, line, paragraph, page
model_id = 'prebuilt-read'      # Has cost implications. Layout more expensive but allows for more features: prebuilt-read, prebuilt-layout

# for query extraction
query_fields = "City, First name, last name"               # string containing comma separated keys to extract
exclude_metadata = True              # if excluded, the resulting table will contain a column per query field (doesn't support ocr metadata like bounding boxes)

# for table extraction


##################### HELPER FUNCTIONS #####################
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

def prepare_query(query_list):
    query_list = query_list.split(',')
    query_list = [q.strip() for q in query_list] # remove leading and trailing whitespace
    query_list = [q.replace(' ', '_') if ' ' in q else q for q in query_list] # replace spaces with underscores
        
    
    # check if query string is regex compatible (azure document intelligence requirement)
    for q in query_list:
        try:
            re.compile(q)
        except re.error:
            raise re.error
        
    return query_list

###################### OCR STRATEGIES #####################
# parent class for the OCR strategies
class OCRStrategy:
    """ Base class for the OCR strategies """
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.kwargs = kwargs

    def parse_ocr_result(self, result) -> pd.DataFrame:
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
        self.model_id = kwargs.get('model_id', 'prebuilt-read')

    def parse_ocr_result(self, result, level):
        # azure doesn't provide results on page level natively
        level = self.lod
        if (level.upper() == "PAGE"):
            lod = "LINE"
        else:
            lod = level.upper()

        for page in result.pages:
            try:
                contains_handwriting = result.styles[0].is_handwritten
            except:
                contains_handwriting = False

            ocr_data = []
            
            # to calculate the average confidence
            if lod != "WORD":
                word_confidences = [word.confidence for word in page.words]
                total_confidence = sum(word_confidences)
                total_words = len(word_confidences)
                average_confidence = total_confidence / total_words if total_words > 0 else 0
                
            # extraction of (natively provided) results 
            if lod == "PARAGRPAH":
                for paragraph_idx, paragraph in enumerate(result.paragraphs):
                    x1, y1, x2, y2, x3, y3, x4, y4 = paragraph.bounding_regions[0].polygon

                    paragrpah_info = {
                        "page": paragraph.bounding_regions[0].page_number,
                        "paragraph": paragraph_idx,
                        "text": paragraph.content,
                        "role": paragraph.role,
                        "bb_x1": x1,
                        "bb_y1": y1,
                        "bb_x2": x2,
                        "bb_y2": y2,
                        "bb_x3": x3,
                        "bb_y3": y3,
                        "bb_x4": x4,
                        "bb_y4": y4,
                        "offset": paragraph.spans[0].offset,
                        "length": paragraph.spans[0].length,
                    }
                    
                    ocr_data.append(paragrpah_info)

            elif lod == "LINE":
                for line_idx, line in enumerate(page.lines):
                    x1, y1, x2, y2, x3, y3, x4, y4 = line.polygon

                    line_info = {
                        "page": page.page_number,
                        "line": line_idx,
                        "text": line.content,
                        "bb_x1": x1,
                        "bb_y1": y1,
                        "bb_x2": x2,
                        "bb_y2": y2,
                        "bb_x3": x3,
                        "bb_y3": y3,
                        "bb_x4": x4,
                        "bb_y4": y4,
                        "offset": line.spans[0].offset,
                        "length": line.spans[0].length,
                    }
                    
                    ocr_data.append(line_info)

            elif lod == "WORD":
                for word in page.words:
                    x1, y1, x2, y2, x3, y3, x4, y4 = word.polygon

                    word_info = {
                        "page": page.page_number,
                        "text": word.content,
                        "confidence": word.confidence,
                        "bb_x1": x1,
                        "bb_y1": y1,
                        "bb_x2": x2,
                        "bb_y2": y2,
                        "bb_x3": x3,
                        "bb_y3": y3,
                        "bb_x4": x4,
                        "bb_y4": y4,
                        "offset": word.spans[0].offset,
                        "length": word.spans[0].length,
                        }
                    
                    ocr_data.append(word_info)
            
            df = pd.DataFrame(ocr_data)

            # in case texts should be aggreagted on page level
            if level.upper() == "PAGE":
                ocr_data = []
                page_info = {
                        "page": page.page_number,
                        "text": "\n ".join(df['text']),
                        "confidence": average_confidence,
                        "contains_handwriting": contains_handwriting,
                        "bb_x1": df["bb_x1"].min(),
                        "bb_y1": df["bb_y1"].min(),
                        "bb_x2": df["bb_x2"].max(),
                        "bb_y2": df["bb_y2"].min(),
                        "bb_x3": df["bb_x3"].max(),
                        "bb_y3": df["bb_y3"].max(),
                        "bb_x4": df["bb_x4"].min(),
                        "bb_x4": df["bb_x4"].max(),
                        }
                ocr_data.append(page_info)
                
                df = pd.DataFrame(ocr_data)

        return df

    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> pd.DataFrame:
        """ Analyze the document and return the result
        
        Returns:
        --------
        parsed_result:
            pd.DataFrame: OCR results
         """
        poller = self.ocr_client.begin_analyze_document( model_id = self.model_id, 
                                                         analyze_request = document,
                                                         content_type="application/octet-stream",
                                                         #locale = self.locale
                                                         )
        result = poller.result()
        parsed_result = self.parse_ocr_result(result = result, level = self.lod)

        if self.model_id == 'prebuilt-read' and self.lod.upper() == 'PARAGRAPH': # 'read' model doesn't provide semantic role, only 'layout' does
            parsed_result = parsed_result.drop(columns=['role'])
        
        return parsed_result

class ExtractForm(OCRStrategy):
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', 'en-US')

    def parse_ocr_result(self, result) -> pd.DataFrame:
        key_value_pairs = result.key_value_pairs
        form_data = []

        for pair in key_value_pairs:
            # get key info
            page_number = pair.key.bounding_regions[0].page_number
            key = pair.key.content
            key_x1, key_y1, key_x2, key_y2, key_x3, key_y3, key_x4, key_y4 = pair.key.bounding_regions[0].polygon
            key_offset = pair.key.spans[0].offset
            key_length = pair.key.spans[0].length
            
            # get value info
            value = pair.get('value', None)
            value_x1 = value_y1 = value_x2 = value_y2 = value_x3 = value_y3 = value_x4 = value_y4 = None
            value_offset = value_length = None

            if value is not None:
                value = value.get('content', None)
                value_x1, value_y1, value_x2, value_y2, value_x3, value_y3, value_x4, value_y4 = pair.value.bounding_regions[0].polygon
                value_offset = pair.value.spans[0].offset
                value_length = pair.value.spans[0].length

            key_value = {
                'page_number': page_number,
                'key': key,
                'value': value,
                'key_x1': key_x1,
                'key_y1': key_y1,
                'key_x2': key_x2,
                'key_y2': key_y2,
                'key_x3': key_x3,
                'key_y3': key_y3,
                'key_x4': key_x4,
                'key_y4': key_y4,
                'key_offset': key_offset,
                'key_length': key_length,
                'value_x1': value_x1,
                'value_y1': value_y1,
                'value_x2': value_x2,
                'value_y2': value_y2,
                'value_x3': value_x3,
                'value_y3': value_y3,
                'value_x4': value_x4,
                'value_y4': value_y4,
                'value_offset': value_offset,
                'value_length': value_length,
            }

            form_data.append(key_value)
        
        df = pd.DataFrame(form_data)

        return df
        
    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> pd.DataFrame:
        poller = self.ocr_client.begin_analyze_document( model_id = "prebuilt-layout", 
                                                        analyze_request = document,
                                                        content_type="application/octet-stream",
                                                        #locale = self.locale,
                                                        features=['keyValuePairs']
                                                        )
        
        result = poller.result()
        parsed_result = self.parse_ocr_result(result = result)

        return parsed_result

class ExtractQuery(OCRStrategy):
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', 'en-US')
        self.query_fields = kwargs.get('query_fields', '')
        self.exclude_metadata = kwargs.get('exclude_metadata', False)

    def parse_ocr_result(self, result, query_fields, exclude_metadata) -> pd.DataFrame:
        query_data = []

        for doc in result.documents:
            for query in query_fields:
                if not exclude_metadata:
                    x1, y1, x2, y2, x3, y3, x4, y4 = doc.fields.get(query).bounding_regions[0].polygon
                    query_info = {
                        'page_number': doc.fields.get(query).bounding_regions[0].page_number,
                        'key': query,
                        'value': doc.fields.get(query).content,
                        'confidence': doc.fields.get(query).confidence,
                        'type': doc.fields.get(query).type,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'x3': x3,
                        'y3': y3,
                        'x4': x4,
                        'y4': y4,
                        'offset': doc.fields.get(query).spans[0].offset,
                        'length': doc.fields.get(query).spans[0].length,
                    }
                    query_data.append(query_info)
                else:
                    query_info = {
                        'key': query,
                        'value': doc.fields.get(query).content,
                    }
                    query_data.append(query_info)

        parsed_result = pd.DataFrame(query_data)

        # if exclude_metadata, transpose results
        if exclude_metadata:
            parsed_result = parsed_result.set_index('key').T

        return parsed_result
        
    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> pd.DataFrame:
        poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                        analyze_request = document,
                                                        content_type = "application/octet-stream",
                                                        #locale = self.locale,
                                                        features = [DocumentAnalysisFeature.QUERY_FIELDS],
                                                        query_fields = self.query_fields,
                                                        )
        
        result = poller.result()
        parsed_result = self.parse_ocr_result(result = result, 
                                              query_fields = self.query_fields, 
                                              exclude_metadata = self.exclude_metadata)

        return parsed_result

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
    def __init__(self, ocr_client: DocumentIntelligenceClient, ocr_type:str, **kwargs):
        self.ocr_client = ocr_client
        self.ocr_type = ocr_type
        self.kwargs = kwargs

        # Define the strategy mapping
        self.strategy_mapping = {
            ('text'): ExtractText,
            ('form'): ExtractForm,
            ('query'): ExtractQuery,
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

form_data = {'file_path': ['data/patient_intake_form_sample.jpg'],
            'filename': ['doc1']}

file_list = pd.DataFrame(form_data)
path_column = 'file_path'



# prepare the query string to the right format
if ocr_type.upper() == 'QUERY':
    query_fields = prepare_query(query_fields)
    print(f'query list: {query_fields}')

# define all possible parameters for the OCR
ocr_params = {
              'ocr_level': lod,
              'locale': locale,
              'model_id': model_id,
              'query_fields': query_fields,
              'exclude_metadata': exclude_metadata,
              }

# initiate the OCR client and processor
ocr_client = DocumentIntelligenceClient(endpoint = ocr_endpoint, 
                                        credential = AzureKeyCredential(ocr_key),
                                        api_version='2023-10-31-preview'
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
        document = io.BytesIO(document.read())

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
        raise e
    
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


    
