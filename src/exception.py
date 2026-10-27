import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error context including script name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = -1
        
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"with message: [{str(error)}]"
    )
    return error_message


class FlightFairException(Exception):
    """
    Custom exception class for FlightFair application with detailed traceback information.
    """
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
