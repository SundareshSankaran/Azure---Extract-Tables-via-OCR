# Imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.ai.documentintelligence.models._models import AnalyzeResult

import pandas as pd
import numpy as np
import functools
import threading
import json
import io
import re
import os
import uuid
from datetime import datetime

# only for development
from dotenv import load_dotenv
load_dotenv()

################### DEFINE PARAMETERS ###################
# azure credentials
ocr_key = os.getenv('OCR_KEY')
ocr_endpoint = os.getenv('OCR_ENDPOINT')

API_VERSION = '2023-10-31-preview'  # default: '2023-10-31-preview'- to lock the API version, in case breaking changes are introduced

# general
ocr_type = 'text'                   # type of OCR: text, form, query, tabel
input_type = 'file'                 # type of input: file, url
input_mode = 'batch'                # single or batch
file_path = 'data/table-test-document.pdf'                      # path to a (single) file
file_list = ''                      # dataframe containing the file names
path_column = ''                    # column that contains the file path
locale = 'en-US'                    # optional, language of the document [ToDo]

n_threads = 30                      # number of threads to use for parallel processing
n_con_retry = 3                     # number of retries if connection fails
retry_delay = 2                     # delay between retries

save_file = True                    # whether to save the json output
output_folder = 'output'            # folder to save the json output  

# for text extraction
lod = 'line'                        # level of detail: word, line, paragraph, page
model_id = 'prebuilt-read'          # Has cost implications. Layout more expensive but allows for more features: prebuilt-read, prebuilt-layout

# for query extraction
query_fields = "City, First name, last name"               # string containing comma separated keys to extract
exclude_metadata = True            # if excluded, the resulting table will contain a column per query field (doesn't support ocr metadata like bounding boxes)

# for table extraction
table_output_format = 'reference'  # how the tables should be returned: map, reference*, table** *reference requires a caslib, **only one table per execution is supported
table_output_caslib = 'work'       # caslib to store the table (only relevant if table_output_format = 'reference')
select_table = False               # whether to select a specific table or all tables (only relevant if table_output_format = 'reference')
tabel_selection_method = 'index'   # how to select the table: size, index (only relevant if table_output_format = 'reference' and selected_table = True)
table_idx = 0                      # index of the table to extract (only relevant if table_output_format = 'table')

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

def prepare_query(query_list: str):
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

    def parse_ocr_result(self, result) -> pd.DataFrame:
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

        if self.model_id == 'prebuilt-read' and self.lod.upper() == 'PARAGRAPH': # 'read' model doesn't provide semantic role, only 'layout' does
            parsed_result = parsed_result.drop(columns=['role'])

        return df

    @retry_on_endpoint_connection_error(max_retries=n_con_retry, delay=retry_delay)
    def analyze_document(self, document) -> AnalyzeResult:
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
        
        return result

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
    def analyze_document(self, document) -> AnalyzeResult:
        poller = self.ocr_client.begin_analyze_document( model_id = "prebuilt-layout", 
                                                        analyze_request = document,
                                                        content_type="application/octet-stream",
                                                        #locale = self.locale,
                                                        features=['keyValuePairs']
                                                        )
        
        result = poller.result()
        
        return result

class ExtractQuery(OCRStrategy):
    def __init__(self, ocr_client, kwargs):
        self.ocr_client = ocr_client
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', 'en-US')
        self.query_fields = kwargs.get('query_fields', '')
        self.exclude_metadata = kwargs.get('exclude_metadata', False)

    def parse_ocr_result(self, result) -> pd.DataFrame:
        query_data = []

        for doc in result.documents:
            for query in self.query_fields:
                if not self.exclude_metadata:
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
    def analyze_document(self, document) -> AnalyzeResult:
        poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                        analyze_request = document,
                                                        content_type = "application/octet-stream",
                                                        #locale = self.locale,
                                                        features = [DocumentAnalysisFeature.QUERY_FIELDS],
                                                        query_fields = self.query_fields,
                                                        )
        
        result = poller.result()

        return result

class ExtractTable(OCRStrategy):
    def __init__(self, ocr_client, kwargs): 
        self.ocr_client = ocr_client
        self.file_location = kwargs.get('file_location', 'local')
        self.locale = kwargs.get('locale', 'en-US')
        self.table_output_format = kwargs.get('table_output_format', 'map')
        self.select_table = kwargs.get('select_table', False)
        self.tabel_selection_method = kwargs.get('tabel_selection_method', 'index')
        self.table_idx = kwargs.get('table_idx', 0)
        self.table_output_caslib = kwargs.get('table_output_caslib', 'work')

    def result_to_dfs(self, result) -> list:
        tables = []
        for table in result.tables:
            table_df = pd.DataFrame(columns=range(table.column_count), index=range(table.row_count))

            for cell in table.cells:
                table_df.iloc[cell.row_index, cell.column_index] = cell.content

            # use the first row as column names
            table_df.columns = table_df.iloc[0]
            table_df = table_df[1:]
            
            tables.append(table_df)
        return tables


    # TABLE PARSING METHODS
    def map_parsing(self, result) -> pd.DataFrame:
        tables = []

        # extract all table data
        for index, table in enumerate(result.tables):
            if self.table_output_format.upper() == 'MAP':
                dict = table.as_dict()
                df = pd.DataFrame.from_dict(dict['cells'])

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
                df['offset'] = df['spans'].apply(lambda x: int(x[0].get('offset')) if x else None)
                df['length'] = df['spans'].apply(lambda x: int(x[0].get('length')) if x else None)

                # drop unnecessary columns
                df.drop(columns=['boundingRegions','spans', 'polygon'], inplace=True)

                table_info = {
                    'table_index': index,
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'cell_count': table.row_count*table.column_count,
                    'table': df
                }

                tables.append(table_info)

        # select specific table (optional)
        if self.select_table:
            if self.tabel_selection_method.upper() == 'INDEX':
                parsed_result = tables[table_idx]['table']
            elif self.tabel_selection_method.upper() == 'SIZE':
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
                'out_caslib': table_output_caslib,
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
            if self.tabel_selection_method.upper() == 'INDEX': # Table with index == table_idx
                parsed_result = tables[table_idx]
            elif self.tabel_selection_method.upper() == 'SIZE': # Table with most cells
                table_most_cells = max(tables, key=lambda x: x.size, default=None)
                parsed_result = table_most_cells if table_most_cells else None

            else:
                raise ValueError(f'Invalid table selection method: {self.tabel_selection_method}')

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
        poller = self.ocr_client.begin_analyze_document(model_id = "prebuilt-layout", 
                                                        analyze_request = document,
                                                        content_type = "application/octet-stream",
                                                        #locale = self.locale,
                                                        )
        
        result = poller.result()

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
        return self.strategy.analyze_document(document)
    
    def parse_ocr_result(self, result:AnalyzeResult) -> pd.DataFrame:
        return self.strategy.parse_ocr_result(result)
    
###################### TEST DATA (FOR DEV) ######################
data = {'file_path': ['data/handwritten-form.jpg','data/letter-example.pdf'],
        'filename': ['doc1', 'doc2']}

form_data = {'file_path': ['data/patient_intake_form_sample.jpg'],
            'filename': ['doc1']}

tabel_data = {'file_path': ['data/table-test-document.pdf'],
            'filename': ['doc1']}

file_list = pd.DataFrame(tabel_data)
path_column = 'file_path'

# create a dataframe with all the file paths of a specified folder not as method yet
def get_file_list(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    # filter out all non-pdf and images
    file_list = [file for file in file_list if file.endswith(('.pdf', '.jpg', '.jpeg', '.png'))]
    return pd.DataFrame({'file_path': file_list})

file_list = get_file_list('data')
print(f'numer of files: {file_list.shape[0]}')

###################### PREP & PRE-CHECKS ######################
if ocr_type.upper() == 'QUERY': # prepare the query string to the right format
    query_fields = prepare_query(query_fields)
    print(f'query list: {query_fields}')

if save_file: # check if output folder should be created (if save_file = True)
    # check if output folder exists
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f'Created output folder {output_folder}!')
        except OSError as e:
            raise OSError(f'Could not create output folder {output_folder}!')
    
    # check if output folder is writable
    if not os.access(output_folder, os.W_OK):
        raise OSError(f'Output folder {output_folder} is not writable!')

if table_output_format.upper() == 'TABLE' and file_list.shape[0] > 1: # if table_output_format = 'table', check if only one row in the file_list
    raise Exception('Only one file is supported if table_output_format = "table"!')

if input_mode.upper() == 'SINGLE': # if input_mode = 'single', create a dataframe with the file path
    file_list = pd.DataFrame({'file_path': [file_path]})
    path_column = 'file_path'

###################### EXECUTION ######################
# define all possible parameters for the OCR
ocr_params = {
              # general
              'ocr_level': lod,
              'locale': locale,
              # for text extraction
              'model_id': model_id,
              # for query extraction
              'query_fields': query_fields,
              'exclude_metadata': exclude_metadata,
              # for table extraction
              'table_output_format': table_output_format,
              'selected_table': select_table,
              'selection_method': tabel_selection_method,
              'table_idx': table_idx,
              'table_output_caslib': table_output_caslib,
              }

# initiate dataframe to store results and status
ocr_results = pd.DataFrame()
status = pd.DataFrame()

# Split file_list into chunks
df_split = np.array_split(file_list, n_threads)

# initiate the OCR client and processor
ocr_client = DocumentIntelligenceClient(endpoint = ocr_endpoint, 
                                        credential = AzureKeyCredential(ocr_key),
                                        api_version=API_VERSION
                                        )

ocr_processor = OCRProcessor(ocr_client = ocr_client, 
                             ocr_type = ocr_type, 
                             **ocr_params
                             )

def process_files(file_list, ocr_processor, path_column):
    # go through every document in the list
    global ocr_results, status

    for _, row in file_list.iterrows():
        print(f'processing file {row[path_column]}')
        done = False
        error_type = ''
        message = ''
        start = datetime.now()

        # perform the OCR
        with open(row[path_column], 'rb') as document:
            document = io.BytesIO(document.read())
        try:
            # analyze the document
            result = ocr_processor.analyze_document(document = document)

            # parse the result
            parsed_result = ocr_processor.parse_ocr_result(result = result)

            # append results to the dataframe
            ocr_results = pd.concat([ocr_results, parsed_result], ignore_index=True)
            ocr_results[path_column]=row[path_column]
            done = True
        except Exception as e:
            error_type = type(e).__name__
            message = str(e)
            print(f'Error: {error_type} - {message}')
            raise e
        
        # Post processing
        if table_output_format.upper() == 'TABLE': # if output_table_format = 'table', drop the path_column
            ocr_results.drop(columns=[path_column], inplace=True)
        if save_file: # if save_file = True, save the azure ocr result as json
            # save the result as json
            try: 
                with open(f'{output_folder}/{row[path_column].split("/")[-1].split(".")[0]}_{ocr_type}.json', 'w') as f:
                    json.dump(result.as_dict(), f)
            except Exception as e:
                error_type = type(e).__name__
                message = str(e)
                print(f'Error: {error_type} - {message}')

        # update the status
        doc_status = {'file': row[path_column],
                    'done': done,
                    'error_type': error_type,
                    'message': message,
                    'start': start,
                    'end': datetime.now(),
                    'duration_seconds': round((datetime.now() - start).total_seconds(), 3)
                    }
        status = pd.concat([status, pd.DataFrame(doc_status, index=[0])], ignore_index=True)

threads = []
for i in range(n_threads):
    paths = df_split[i]
    thread = threading.Thread(target=process_files, args=(paths, ocr_processor, path_column))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()


# print & save the results (dev only)
print(f'Successfully processed {status["done"].sum()} / {status.shape[0]} files!')
ocr_results.to_csv('ocr_results.csv')
status.to_csv('ocr_status.csv')

 



    
