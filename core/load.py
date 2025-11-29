from data.vectorstore_manager import create_vectorstore
import tempfile

def process_file(path=None, file_obj=None, lib=None):
    
    if file_obj is not None:
        suffix = f".{type}" if type else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        path_to_use = tmp_path
    else:
        path_to_use = path

    create_vectorstore(input_path=path_to_use, lib_name=lib)
