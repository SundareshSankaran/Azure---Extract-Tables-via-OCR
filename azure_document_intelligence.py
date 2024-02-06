# Imports
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature
    from azure.ai.documentintelligence.models._models import AnalyzeResult, AnalyzeDocumentRequest

    import pandas as pd
    import numpy as np
    import functools
    import requests
    import threading
    import json
    import io
    import re
    import os
    import uuid
    from urllib.parse import urlparse
    from datetime import datetime
except ImportError as e:
    print(f'ImportError - {e}. Please install the required packages!')
    exit()

# only for development
from dotenv import load_dotenv
load_dotenv()

################### DEFINE PARAMETERS ###################
# azure credentials
azure_key = str(os.getenv('OCR_KEY'))
azure_endpoint = str(os.getenv('OCR_ENDPOINT'))
local_ocr = bool(0)                             # whether to use a locally deployed document intelligence container, default = False
local_ocr_endpoint = str('http://localhost:5000') # endpoint of the locally deployed document intelligence container

SERVICE_VERSION = '4.0'                         # 4.0 is in preview. Local containers are only supported in 3.0 (GA) thus far
API_VERSION = '2023-10-31-preview'              # default: '2023-10-31-preview'- to lock the API version, in case breaking change are introduced

# general
ocr_type = str('query')                          # type of OCR: text, form, query, table
input_type = str('file')                        # type of input: file, url 
input_mode = str('batch')                       # single or batch
file_path = str('data/table-test-document.pdf') # path to a (single) file
input_table_name = None                         # name of table containing the file paths
path_column = str('')                           # column that contains the file path
locale = str('en')                              # optional, language of the document. Support-list: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/language-support-ocr?view=doc-intel-4.0.0&tabs=read-hand%2Clayout-print%2Cgeneral

n_threads = int(32)                             # number of threads to use for parallel processing
n_con_retry = int(3)                            # number of retries if connection fails
retry_delay = int(2)                            # delay between retries
output_status_table = bool(1)                   # whether to output the status table

save_json = bool(False)                         # whether to save the json output
json_output_folder = str('output')              # folder to save the json output

# for text extraction
text_granularity = str('line')               # level of detail: word, line, paragraph, page, document
model_id = str('prebuilt-layout')                # Has cost implications. Layout more expensive but allows for more features: prebuilt-read, prebuilt-layout

# for query extraction
query_fields = str("City, First name")          # string containing comma separated keys to extract
query_exclude_metadata = bool(True)             # if excluded, the resulting table will contain a column per query field (doesn't support ocr metadata like bounding boxes)

# for table extraction
table_output_format = str('table')                # how the tables should be returned: map, reference*, table** *reference requires a cas
table_output_library = str('work')              # caslib to store the table (only relevant if table_output_format = 'reference')
select_table = bool(False)                      # whether to select a specific table or all tables (only relevant if table_output_format = 'reference')
table_selection_method = str('index')           # how to select the table: size, index (only relevant if table_output_format = 'reference' and selected_table = True)
table_selection_idx = int(0)                    # index of the table to extract (only relevant if table_output_format = 'table')

##################### HELPER FUNCTIONS #####################
def retry_on_endpoint_connection_error(max_retries=3, delay=2):
    """
    This is a decorator function that allows a function to retry execution when an EndpointConnectionError occurs.

    Parameters:
    -----------
    max_retries (int): 
        The maximum number of retries if an EndpointConnectionError occurs. Default is 3.
    delay (int): 
        The delay (in seconds) between retries. Default is 2.

    Returns:
    wrapper function: 
        The decorated function that includes retry logic.
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

def prepare_query(query_list: str):
    """ Parse the query string to a list of query keys

    Parameters:
    -----------
    query_list:
        str: comma separated string of queries
    
    Returns:
    --------
    query_list:
        list(str): list of queries
    """
    query_list = query_list.split(',')
    query_list = [q.strip() for q in query_list] # remove leading and trailing whitespace
    query_list = [q.replace(' ', '_') if ' ' in q else q for q in query_list] # replace spaces with underscores
        
    
    # check if query string is regex compatible (azure document intelligence requirement)
    for q in query_list:
        try:
            re.compile(q)
        except re.error:
            ValueError(f'Query string {q} is not regex compatible!')
        
    return query_list


    """ Check if a url is valid

    Parameters:
    -----------
    url:
        str: url to check
    
    Returns:
    --------
    bool: True if url is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

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
        self.local_ocr = kwargs.get('local_ocr', False)
        self.input_type = kwargs.get('input_type', 'file')
        self.text_granularity = kwargs.get('text_granularity', 'line')
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', '')
        self.model_id = kwargs.get('model_id', 'prebuilt-read')

        if self.local_ocr:
            self.endpoint = kwargs.get('endpoint', 'http://localhost:5000')

    def parse_ocr_result(self,result) -> pd.DataFrame:
        parsed_result = pd.DataFrame()

        # set the text granularity
        level = self.text_granularity
        if (level.upper() == 'PAGE'):
            self.text_granularity = "LINE"
        else:
            self.text_granularity = level.upper()
        
        if self.text_granularity == "DOCUMENT":
            ocr_data = []
            
            # check if the document contains handwriting
            try:
                contains_handwriting = result['styles'][0]['isHandwritten']
            except:
                contains_handwriting = False

            document_info = {
                "text": result['content'],
                "contains_handwriting": contains_handwriting,
                }
            ocr_data.append(document_info)
            df = pd.DataFrame(ocr_data)
            parsed_result = pd.concat([parsed_result, df], ignore_index=True)

        elif self.text_granularity == "PARAGRAPH":
            ocr_data = []
            print('paragraph')
            for paragraph_idx, paragraph in enumerate(result['paragraphs']):
                x1, y1, x2, y2, x3, y3, x4, y4 = paragraph['boundingRegions'][0]['polygon']

                try: 
                    role = paragraph['role']
                except:
                    role = ''

                paragrpah_info = {
                    "page": paragraph['boundingRegions'][0]['pageNumber'],
                    "paragraph": paragraph_idx,
                    "role": role,
                    "text": paragraph['content'], 
                    "bb_x1": x1,
                    "bb_y1": y1,
                    "bb_x2": x2,
                    "bb_y2": y2,
                    "bb_x3": x3,
                    "bb_y3": y3,
                    "bb_x4": x4,
                    "bb_y4": y4,
                    "offset": paragraph['spans'][0]['offset'],
                    "length": paragraph['spans'][0]['length'],
                }
                
                ocr_data.append(paragrpah_info)
            df = pd.DataFrame(ocr_data)
            parsed_result = pd.concat([parsed_result, df], ignore_index=True)
        else:
            for page in result['pages']:
                ocr_data = []
                
                # to calculate the average confidence
                if self.text_granularity != "WORD":
                    word_confidences = [word['confidence'] for word in page['words']]
                    total_confidence = sum(word_confidences)
                    total_words = len(word_confidences)
                    average_confidence = total_confidence / total_words if total_words > 0 else 0

                # extraction on line level
                if self.text_granularity == "LINE":
                    for line_idx, line in enumerate(page['lines']):
                        x1, y1, x2, y2, x3, y3, x4, y4 = line['polygon']

                        line_info = {
                            "page": page['pageNumber'],
                            "line": line_idx,
                            "text": line['content'],
                            "bb_x1": x1,
                            "bb_y1": y1,
                            "bb_x2": x2,
                            "bb_y2": y2,
                            "bb_x3": x3,
                            "bb_y3": y3,
                            "bb_x4": x4,
                            "bb_y4": y4,
                            "offset": line['spans'][0]['offset'],
                            "length": line['spans'][0]['length'],
                        }
                        
                        ocr_data.append(line_info)

                # extraction on word level
                elif self.text_granularity == "WORD":
                    for word in page['words']:
                        x1, y1, x2, y2, x3, y3, x4, y4 = word['polygon']

                        word_info = {
                            "page": page['pageNumber'],
                            "text": word['content'],
                            "confidence": word['confidence'],
                            "bb_x1": x1,
                            "bb_y1": y1,
                            "bb_x2": x2,
                            "bb_y2": y2,
                            "bb_x3": x3,
                            "bb_y3": y3,
                            "bb_x4": x4,
                            "bb_y4": y4,
                            "offset": word['span']['offset'],
                            "length": word['span']['length'],
                            }
                        
                        ocr_data.append(word_info)
                
                df = pd.DataFrame(ocr_data)

                # aggregation on page level
                if level.upper() == "PAGE":
                    ocr_data = []
                    page_info = {
                            "page": page['pageNumber'],
                            "text": "\n ".join(df['text']),
                            "avg_confidence": average_confidence
                            }
                    ocr_data.append(page_info)
                    
                    df = pd.DataFrame(ocr_data)

                parsed_result = pd.concat([parsed_result, df], ignore_index=True)

        if self.model_id == 'prebuilt-read' and self.text_granularity.upper() == 'PARAGRAPH': # 'read' model doesn't provide semantic role, only 'layout' does
            parsed_result = parsed_result.drop(columns=['role'])

        return parsed_result

    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> AnalyzeResult:
        """ Analyze the document and return the result

        Parameters:
        -----------
        document:
            io.BytesIO|str: document or url to document to analyze
        
        Returns:
        --------
        parsed_result:
            pd.DataFrame: OCR results
         """
        if not self.local_ocr:
            if self.input_type.upper() == 'FILE':
                poller = self.ocr_client.begin_analyze_document(model_id = self.model_id, 
                                                                analyze_request = document,
                                                                content_type="application/octet-stream",
                                                                locale = self.locale
                                                                )
            elif self.input_type.upper() == 'URL':
                poller = self.ocr_client.begin_analyze_document(model_id = self.model_id, 
                                                                analyze_request = AnalyzeDocumentRequest(url_source=document),
                                                                locale = self.locale
                                                                )
            
            result = poller.result()
        else:
            url = f"{self.endpoint}/formrecognizer/documentModels/prebuilt-read:syncAnalyze?api-version=2022-08-31"
            headers = {
                'accept': '*/*',
                'Content-Type': 'application/octet-stream',
            }
            response = requests.post(url, headers=headers, data=document)
            response_json = json.loads(response.text)
            result = response_json['analyzeResult']

        return result

class ExtractForm(OCRStrategy):
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.local_ocr = kwargs.get('local_ocr', False)
        self.input_type = kwargs.get('input_type', 'file')
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', '')

        if self.local_ocr:
            self.endpoint = kwargs.get('endpoint', 'http://localhost:5000')

    def parse_ocr_result(self, result) -> pd.DataFrame:
        key_value_pairs = result['keyValuePairs']
        form_data = []

        for pair in key_value_pairs:
            page_number = pair['key']['boundingRegions'][0]['pageNumber']
            key = pair['key']['content']
            key_x1, key_y1, key_x2, key_y2, key_x3, key_y3, key_x4, key_y4 = pair['key']['boundingRegions'][0]['polygon']
            key_offset = pair['key']['spans'][0]['offset']
            key_length = pair['key']['spans'][0]['length']

            try:
                value = pair['value']['content']
                value_x1, value_y1, value_x2, value_y2, value_x3, value_y3, value_x4, value_y4 = pair['value']['boundingRegions'][0]['polygon']
                value_offset = pair['value']['spans'][0]['offset']
                value_length = pair['value']['spans'][0]['length']
            except KeyError as e:
                value_x1 = value_y1 = value_x2 = value_y2 = value_x3 = value_y3 = value_x4 = value_y4 = None
                value_offset = value_length = None
                value = None

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
    def analyze_document(self, document) -> AnalyzeResult:
        
        if not local_ocr:
            if self.input_type.upper() == 'FILE':
                poller = self.ocr_client.begin_analyze_document( model_id = "prebuilt-layout", 
                                                            analyze_request = document,
                                                            content_type="application/octet-stream",
                                                            locale = self.locale,
                                                            features=['keyValuePairs']
                                                            )
            
            elif self.input_type.upper() == 'URL':
                poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                                analyze_request = AnalyzeDocumentRequest(url_source=document),
                                                                locale = self.locale,
                                                                features=['keyValuePairs']
                                                                )
            
            result = poller.result()
        else:
            url = f"{self.endpoint}/formrecognizer/documentModels/prebuilt-document:syncAnalyze?api-version=2022-08-31"
            headers = {
                'accept': '*/*',
                'Content-Type': 'application/octet-stream',
            }
            response = requests.post(url, headers=headers, data=document)
            response_json = json.loads(response.text)
            result = response_json['analyzeResult']
        
        return result

class ExtractQuery(OCRStrategy):
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.input_type = kwargs.get('input_type', 'file')
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', '')
        self.query_fields = kwargs.get('query_fields', '')
        self.query_exclude_metadata = kwargs.get('query_exclude_metadata', False)

    def parse_ocr_result(self, result) -> pd.DataFrame:
        query_data = []

        for doc in result['documents']:
            for query in self.query_fields:
                if not self.query_exclude_metadata:
                    x1, y1, x2, y2, x3, y3, x4, y4 = doc['fields'][query]['boundingRegions'][0]['polygon']
                    query_info = {
                        'page_number': doc['fields'][query]['boundingRegions'][0]['pageNumber'],
                        'key': query,
                        'value': doc['fields'][query]['content'],
                        'confidence': doc['fields'][query]['confidence'],
                        'type': doc['fields'][query]['type'],
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'x3': x3,
                        'y3': y3,
                        'x4': x4,
                        'y4': y4,
                        'offset': doc['fields'][query]['spans'][0]['offset'],
                        'length': doc['fields'][query]['spans'][0]['length'],
                    }
                    query_data.append(query_info)
                else:
                    query_info = {
                        'key': query,
                        'value': doc['fields'][query]['content'],
                    }
                    query_data.append(query_info)

        parsed_result = pd.DataFrame(query_data)

        # if query_exclude_metadata, transpose results
        if query_exclude_metadata:
            parsed_result = parsed_result.set_index('key').T

        return parsed_result
        
    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> AnalyzeResult:
        if self.input_type.upper() == 'FILE':
            poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                        analyze_request = document,
                                                        content_type = "application/octet-stream",
                                                        locale = self.locale,
                                                        features = [DocumentAnalysisFeature.QUERY_FIELDS],
                                                        query_fields = self.query_fields,
                                                        )
        
        elif self.input_type.upper() == 'URL':
            poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                            analyze_request = AnalyzeDocumentRequest(url_source=document),
                                                            locale = self.locale,
                                                            features = [DocumentAnalysisFeature.QUERY_FIELDS],
                                                            query_fields = self.query_fields,
                                                            )
        result = poller.result()

        return result

class ExtractTable(OCRStrategy):
    def __init__(self, ocr_client, kwargs): 
        self.ocr_client = ocr_client
        self.local_ocr = kwargs.get('local_ocr', False)
        self.input_type = kwargs.get('input_type', 'file')
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', '')
        self.table_output_format = kwargs.get('table_output_format', 'map')
        self.select_table = kwargs.get('select_table', False)
        self.table_selection_method = kwargs.get('table_selection_method', 'index')
        self.table_selection_idx = kwargs.get('table_selection_idx', 0)
        self.table_output_caslib = kwargs.get('table_output_caslib', 'work')

        if self.local_ocr:
            self.endpoint = kwargs.get('endpoint', 'http://localhost:5000')

    def result_to_dfs(self, result) -> list:
        tables = []
        for table in result['tables']:
            table_df = pd.DataFrame(columns=range(table['columnCount']), index=range(table['rowCount']))

            for cell in table['cells']:
                table_df.iloc[cell['rowIndex'], cell['columnIndex']] = cell['content']

            # use the first row as column names
            table_df.columns = table_df.iloc[0]
            table_df = table_df[1:]
            
            tables.append(table_df)
        return tables

    # TABLE PARSING METHODS
    def map_parsing(self, result) -> pd.DataFrame:
        tables = []

        # extract all table data
        for index, table in enumerate(result['tables']):
            if self.table_output_format.upper() == 'MAP':

                if not isinstance(table, dict):
                    table = table.as_dict()

                df = pd.DataFrame.from_dict(table['cells'])

                # extract page_number and polygon coordinates
                df['page'] = df['boundingRegions'].apply(lambda x: x[0]['pageNumber'])
                df['table_index'] = index
                df['polygon'] = df['boundingRegions'].apply(lambda x: x[0]['polygon'])

                # extract polygon coordinates
                df['x1'] = df['polygon'].apply(lambda x: x[0])
                df['y1'] = df['polygon'].apply(lambda x: x[1])
                df['x2'] = df['polygon'].apply(lambda x: x[2])
                df['y2'] = df['polygon'].apply(lambda x: x[3])
                df['x3'] = df['polygon'].apply(lambda x: x[4])
                df['y3'] = df['polygon'].apply(lambda x: x[5])
                df['x4'] = df['polygon'].apply(lambda x: x[6])
                df['y4'] = df['polygon'].apply(lambda x: x[7])

                # extract offset and length
                df['offset'] = df['spans'].apply(lambda x: int(x[0]['offset']) if x else None)
                df['length'] = df['spans'].apply(lambda x: int(x[0]['length']) if x else None)

                # drop unnecessary columns
                df.drop(columns=['boundingRegions','spans', 'polygon'], inplace=True)

                table_info = {
                    'table_index': index,
                    'row_count': table['rowCount'],
                    'column_count': table['columnCount'],
                    'cell_count': table['rowCount']*table['columnCount'],
                    'table': df
                }

                tables.append(table_info)

        # select specific table (optional)
        if self.select_table:
            if self.table_selection_method.upper() == 'INDEX':
                parsed_result = tables[table_selection_idx]['table']
            elif self.table_selection_method.upper() == 'SIZE':
                # Find the entry with the highest cell_count using max function
                table_most_cells = max(tables, key=lambda x: x['cell_count'], default=None)
                parsed_result = table_most_cells['table'] if table_most_cells else None

        else:
            # combine all extracted tables (only works for output type 'map')
            parsed_result = pd.concat([table['table'] for table in tables], ignore_index=True)

        return parsed_result

    def reference_parsing(self, result) -> pd.DataFrame: # TODO
        tables = self.result_to_dfs(result)
        table_info = []

        for table in tables:
            reference = uuid.uuid4()
            reference = re.sub(r'^\w{3}', 'tbl_', str(reference))
            reference = reference.replace('-', '')

            # save table to caslib
            try: 
                print(f'Save table {reference} to caslib {self.table_output_caslib}')
            except Exception as e:
                print(f'Failed to save table {reference} to caslib {self.table_output_caslib}')
                raise e
            
            table_info.append({
                'out_library': self.table_output_caslib,
                'table_reference': reference,
                'row_count': table.shape[0],
                'column_count': table.shape[1],
            })

        return pd.DataFrame(table_info)

    def table_parsing(self, result) -> pd.DataFrame: #TODO
        tables = self.result_to_dfs(result)
        self.select_table = True

        # select specific table 
        if self.select_table:
            if self.table_selection_method.upper() == 'INDEX': # Table with index == table_selection_idx
                parsed_result = tables[table_selection_idx]
            elif self.table_selection_method.upper() == 'SIZE': # Table with most cells
                table_most_cells = max(tables, key=lambda x: x.size, default=None)
                try:
                    parsed_result = table_most_cells
                except:
                    parsed_result = None

            else:
                raise ValueError(f'Invalid table selection method: {self.table_selection_method}')

        return parsed_result

    # TABLE PARSING METHODS MAPPING
    parsing_methods = {
        'MAP': map_parsing,
        'REFERENCE': reference_parsing,
        'TABLE': table_parsing
    }

    def parse_ocr_result(self, result) -> pd.DataFrame:
        # call one of the parsing methods depending on the output format
        parsing_method = table_output_format.upper()
        parsed_result = self.parsing_methods.get(parsing_method)(self,result = result)

        return parsed_result

    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> AnalyzeResult:
        if not self.local_ocr:   
            if self.input_type.upper() == 'FILE':
                poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                            analyze_request = document,
                                                            content_type = "application/octet-stream",
                                                            locale = self.locale,
                                                            )
            
            elif self.input_type.upper() == 'URL':
                poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                                analyze_request = AnalyzeDocumentRequest(url_source=document),
                                                                locale = self.locale
                                                                )

            result = poller.result()      
        else:
            url = f"{self.endpoint}/formrecognizer/documentModels/prebuilt-document:syncAnalyze?api-version=2022-08-31"
            headers = {
                'accept': '*/*',
                'Content-Type': 'application/octet-stream',
            }
            response = requests.post(url, headers=headers, data=document)
            response_json = json.loads(response.text)
            result = response_json['analyzeResult']
        
        return result

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

    def analyze_document(self, document:io.BytesIO|str) -> AnalyzeResult:
        """ Analyze the document and return the result
        
        Parameters:
        -----------
        document:
            io.BytesIO|str: document to analyze
            
        Returns:
        --------
        result:
            AnalyzeResult: OCR results"""
        return self.strategy.analyze_document(document)
    
    def parse_ocr_result(self, result:AnalyzeResult) -> pd.DataFrame:
        """ Parse the OCR result and return the result

        Parameters:
        -----------
        result:
            AnalyzeResult: OCR results

        Returns:
        --------
        parsed_result:
            pd.DataFrame: parsed OCR results
        """
        return self.strategy.parse_ocr_result(result)
    
###################### TEST DATA (FOR DEV) ######################
data = {'file_path': ['data/table-test-document.pdf'],
        'filename': ['doc1']}

form_data = {'file_path': ['data/patient_intake_form_sample.jpg'],
            'filename': ['doc1']}

tabel_data = {'file_path': ['data/table-test-document.pdf'],
            'filename': ['doc1']}

url_data = {'file_path': ['https://raw.githubusercontent.com/Azure/azure-sdk-for-python/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_forms/receipt/contoso-receipt.png'],
            'filename': ['doc1']}

file_list = pd.DataFrame(form_data)
path_column = 'file_path'

# create a dataframe with all the file paths of a specified folder not as method yet
def get_file_list(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    # filter out all non-pdf and images
    file_list = [file for file in file_list if file.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.docx'))]
    return pd.DataFrame({'file_path': file_list})

#file_list = get_file_list('data')
print(f'numer of files: {file_list.shape[0]}')

###################### PREPARATION ######################
if input_mode.upper() == 'BATCH':                       # When input_mode = 'batch' try to load the file list using the input_table_name
    try:
        #file_list = SAS.sd2df(input_table_name)
        pass
    except Exception as e:
        #SAS.logMessage('No input table was provided!}', 'error')
        pass
else:
	file_list = ''
if ocr_type.upper() == 'QUERY':                         # prepare the query string to the right format
    try:
        query_fields = prepare_query(query_fields)
    except ValueError as e:
        print(f'REGEX ERROR: {e}')
        exit()
    except Exception as e:
        print(f'ERROR: {e}')
        exit()  
if input_mode.upper() == 'SINGLE':                      # if input_mode = 'single', create a dataframe with the file path
    file_list = pd.DataFrame({'file_path': [file_path]})
    path_column = 'file_path'
if save_json:                                           # check if output folder should be created (if save_json = True)
    # check if output folder exists
    if not os.path.exists(json_output_folder):
        try:
            os.makedirs(json_output_folder)
            print(f'Created output folder {json_output_folder}!')
        except OSError as e:
            raise OSError(f'OSError - Could not create output folder {json_output_folder}!')
            exit()
    
    # check if output folder is writable
    if not os.access(json_output_folder, os.W_OK):
        raise OSError(f'OSError - Output folder {json_output_folder} is not writable!')
        exit()
if local_ocr:                                           # check if local ocr container is running and reachable
    for check in ['status', 'ready', 'containerliveness']:
        url = f'{local_ocr_endpoint}/{check}'
        headers = {
            'accept': '*/*',
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f'Local OCR Container is not running or cant be reached! {check}: {response.status_code}')
            exit()
    
    print('Local OCR Container is running!')
                                                                           
###################### PRE-CHECKS ######################
if input_mode.upper() == 'SINGLE':                                      # When input type is 'file' check if the file is located on the server not SAS Content
	try:
		ocr_document_path = file_path.split(':', 1)[1]
	except Exception as e:
		#SAS.logMessage("Please select a valid path. Files have to be located on SAS Server (not SAS Content)!", 'error')
		exit()
if ocr_type.upper() == 'TABLE' and table_output_format.upper() == 'TABLE' and file_list.shape[0] > 1:   # if table_output_format = 'table', check if only one row in the file_list
    raise ValueError('Only one file is supported if table_output_format = "table"!')
    exit()
if input_mode.upper() == 'BATCH' and file_list.shape[0] < 1:            # if input_mode = 'batch' and input_type = 'file', check if the file list is not empty
    raise ValueError('Provided file list is empty!')
    exit()
if local_ocr:
    if ocr_type.upper() == 'QUERY':
        raise ValueError('Local OCR does not support query extraction!')
        # SAS.logMessage('Local OCR Container does not support query extraction!', 'error')
        exit()

###################### EXECUTION ######################
# define all possible parameters for the OCR
ocr_params = {
              # general
              'locale': locale,
              'input_type': input_type,
              'local_ocr': local_ocr,
              # for text extraction
              'text_granularity': text_granularity,
              'model_id': model_id,
              # for query extraction
              'query_fields': query_fields,
              'query_exclude_metadata': query_exclude_metadata,
              # for table extraction
              'table_output_format': table_output_format,
              'selected_table': select_table,
              'selection_method': table_selection_method,
              'table_selection_idx': table_selection_idx,
              'table_output_caslib': table_output_library,
              }

# initiate dataframe to store results and status
ocr_results = pd.DataFrame()
status = pd.DataFrame()

# initiate the OCR client and processor
if not local_ocr:
    ocr_client = DocumentIntelligenceClient(endpoint = azure_endpoint, 
                                        credential = AzureKeyCredential(azure_key),
                                        api_version = API_VERSION
                                        )
else:
    ocr_client = {'endpoint': local_ocr_endpoint}

ocr_processor = OCRProcessor(ocr_client = ocr_client, 
                             ocr_type = ocr_type, 
                             **ocr_params
                             )

def process_files(file_list, ocr_processor, path_column):
    """ Process the files in the file_list using the ocr_processor

    Parameters:
    -----------
    file_list:
        pd.DataFrame: dataframe containing the file paths
    ocr_processor:
        OCRProcessor: OCR processor
    path_column:
        str: column that contains the file path
    """
    # go through every document in the list
    global ocr_results, status

    for _, row in file_list.iterrows():
        print(f'processing file {row[path_column]}')
        done = False
        n_rows = 0
        error_type = ''
        message = ''
        start = datetime.now()

        # perform the OCR
        if input_type.upper() == 'FILE':
            with open(row[path_column], 'rb') as document:
                document = io.BytesIO(document.read())
        elif input_type.upper() == 'URL':
            document = row[path_column]
        else:
            raise ValueError(f'Invalid input type: {input_type}!')
        
        try:
            # run ocr processing on the document
            result = ocr_processor.analyze_document(document = document)

            # parse the ocr result
            parsed_result = ocr_processor.parse_ocr_result(result = result)

            # add the file path to the result
            if not parsed_result.empty:
                parsed_result[path_column] = row[path_column]

            # append result to the overall result table
            if not parsed_result.empty:
                ocr_results = pd.concat([ocr_results, parsed_result], ignore_index=True)
                n_rows = parsed_result.shape[0]

            done = True

        except Exception as e:
            error_type = type(e).__name__
            message = str(e)
            print(f'Warning: {error_type} - {message} - for {row[path_column]}')
        
        # Post processing
        if ocr_type.upper() == 'TABLE' and table_output_format.upper() == 'TABLE': # if output_table_format = 'table', drop the path_column
            ocr_results.drop(columns=[path_column], inplace=True)

        if save_json: # if save_json = True, save the azure ocr result as json
            try: 
                with open(f'{json_output_folder}/{row[path_column].split("/")[-1].split(".")[0]}_{ocr_type}.json', 'w') as f:
                    json.dump(result.as_dict(), f)
            except Exception as e:
                error_type = type(e).__name__
                message = str(e)
                print(f'Warning: {error_type} - {message} for {row[path_column]}')

        # update the status
        doc_status = {'file': row[path_column],
                    'done': done,
                    'num_rows': n_rows,
                    'error_type': error_type,
                    'message': message,
                    'start': start,
                    'end': datetime.now(),
                    'duration_seconds': round((datetime.now() - start).total_seconds(), 3)
                    }
        status = pd.concat([status, pd.DataFrame(doc_status, index=[0])], ignore_index=True)

# Parallel processing of the files
df_split = np.array_split(file_list, n_threads)
threads = []

if file_list.shape[0] < n_threads:
    n_threads = file_list.shape[0]

for i in range(n_threads):
    paths = df_split[i]
    thread = threading.Thread(target=process_files, args=(paths, ocr_processor, path_column))
    threads.append(thread)
    thread.start()

    print(f'INFO - Started thread {i+1} of {n_threads}!')

# Wait for all threads to complete
for index, thread in enumerate(threads):
    thread.join()
    print(f'INFO - Thread {index+1} of {n_threads} completed!')

print(f'FINISHED - Successfully processed {status["done"].sum()} / {status.shape[0]} files!')

# Output the results
#SAS.df2sd(ocr_results, SAS.symget("_output1"))
if output_status_table:
    #SAS.df2sd(status, SAS.symget("_output2"))
    pass

# print & save the results (dev only)
ocr_results.to_csv('ocr_results.csv')
status.to_csv('ocr_status.csv')